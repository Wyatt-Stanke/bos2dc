"""End-to-end orchestration: compiled feeds -> graph -> routes -> report data."""

from __future__ import annotations

import datetime as dt
import logging
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config
from .catalog import FeedChoice, read_manifest
from .geo import distance_to_polyline_m, project
from .flex import read_zones
from .graph import Graph, GraphParams, add_flex, attach_places, build_graph
from .gtfs import CompiledFeed, compile_cached
from .itinerary import Simulator, serving_options
from .search import CostParams, Route, Searcher

log = logging.getLogger(__name__)


@dataclass
class Options:
    data_dir: Path
    date: dt.date
    graph: GraphParams = field(default_factory=GraphParams)
    cost: CostParams = field(default_factory=CostParams)
    exclude_agency: list[str] = field(default_factory=list)
    exclude_feed: list[str] = field(default_factory=list)
    max_stale_days: int | None = None
    buffer_km: float = config.CORRIDOR_BUFFER_KM
    workers: int = 3
    max_walk_m: float | None = None  # None: gap_radius, raised automatically if needed
    auto_walk: bool = True
    use_flex: bool = False
    flex_files: list[str] = field(default_factory=list)
    # "roads": locality by road type (OpenStreetMap, see roads.py);
    # "stops": by stop density.
    locality: str = "roads"


@dataclass
class Result:
    graph: Graph
    manifest: dict[str, FeedChoice]
    best_threshold: float
    routes: list[Route]  # labelled: recommended first, then most-local, then trade-offs
    profiles: list[list[tuple[int, int]]]
    simulators: list[Simulator]
    max_walk_m: float = 0.0
    walk_note: str = ""


def next_weekday(today: dt.date, weekday: int) -> dt.date:
    return today + dt.timedelta(days=(weekday - today.weekday()) % 7)


def _compile_one(args):
    path, feed_id, provider, date, cache = args
    try:
        return compile_cached(path, feed_id, provider, date, cache)
    except Exception as e:  # noqa: BLE001 - one broken feed must not sink the run
        log.warning("compile %s failed: %s", feed_id, e)
        return None


def load_feeds(opts: Options) -> tuple[list[CompiledFeed], dict[str, FeedChoice]]:
    manifest = {c.feed_id: c for c in read_manifest(opts.data_dir / "manifest.json")}
    jobs = []
    provider_re = config.compile_patterns(config.EXCLUDED_PROVIDER_PATTERNS)
    for c in manifest.values():
        # Hand-picked atlas feeds bypass the provider filter (UConn's feed
        # carries Windham Region Transit District's public routes).
        if c.feed_id in opts.exclude_feed or not c.path or (c.source != "atlas-extra" and provider_re.search(c.provider)):
            continue
        if opts.max_stale_days is not None and c.stale and c.service_end:
            age = (opts.date - dt.date.fromisoformat(c.service_end)).days
            if age > opts.max_stale_days:
                log.info("skip %s: stale by %d days", c.feed_id, age)
                continue
        jobs.append((c.path, c.feed_id, c.provider, opts.date, opts.data_dir / "compiled"))
    # Largest archives first so the pool stays busy.
    jobs.sort(key=lambda j: -Path(j[0]).stat().st_size)
    feeds = []
    with ProcessPoolExecutor(max_workers=opts.workers) as pool:
        for f in pool.map(_compile_one, jobs):
            if f is None:
                continue
            lat = np.concatenate([f.stops["lat"].to_numpy()] + [r[:, 1] for z in f.flex_zones for r in z.rings])
            lon = np.concatenate([f.stops["lon"].to_numpy()] + [r[:, 0] for z in f.flex_zones for r in z.rings])
            if not len(lat) or distance_to_polyline_m(lat, lon, config.CORRIDOR_SPINE).min() > opts.buffer_km * 1000:
                continue
            feeds.append(f)
    feeds = _apply_agency_exclusions(feeds, opts.exclude_agency)
    log.info("compiled %d feeds", len(feeds))
    return feeds, manifest


def _apply_agency_exclusions(feeds: list[CompiledFeed], patterns: list[str]) -> list[CompiledFeed]:
    if not patterns:
        return feeds
    rx = re.compile("|".join(patterns), re.IGNORECASE)
    out = []
    for f in feeds:
        if rx.search(f.provider) or rx.search(f.feed_id):
            continue
        f.patterns = [p for p in f.patterns if not rx.search(p.agency) and not rx.search(f"{p.agency} {p.route_name}")]
        if f.patterns:
            out.append(f)
    return out


def road_index_path(data_dir: Path) -> Path:
    return data_dir / "osm" / "roads.npz"


def build(opts: Options, origin=config.SOUTH_STATION, destination=config.UNION_STATION) -> tuple[Graph, dict[str, FeedChoice]]:
    feeds, manifest = load_feeds(opts)
    by_road = None
    if opts.locality == "roads":
        from .roads import classify_feeds

        index = road_index_path(opts.data_dir)
        if not index.exists():
            raise RuntimeError(f"no road index at {index}: run `bos2dc roads` first, or use --locality stops")
        by_road = classify_feeds(feeds, {fid: c.path for fid, c in manifest.items()}, index, opts.data_dir / "roads", opts.workers)
    g = build_graph(feeds, opts.graph, roads=by_road)
    g = attach_places(g, origin, destination)
    zones = []
    if opts.use_flex:
        zones = [z for f in feeds for z in f.flex_zones] + read_zones(opts.data_dir / "flex_zones.geojson")
        for path in opts.flex_files:
            zones += read_zones(Path(path))
    return add_flex(g, zones, opts.date), manifest


def run(opts: Options) -> Result:
    g, manifest = build(opts)

    max_walk = opts.max_walk_m if opts.max_walk_m is not None else opts.graph.gap_radius_m
    walk_note = ""
    s = Searcher(g, opts.cost, max_walk)
    if not s.reachable(float(s.levels[0])):
        need = s.required_walk() if opts.auto_walk else None
        if need is None:
            raise RuntimeError("Union Station is unreachable from South Station with the loaded feeds.\n" + gap_report(g, max_walk))
        walk_note = (f"No route exists with walks under {max_walk / 1000:.1f} km; the shortest possible longest walk is "
                     f"{need / 1000:.2f} km (straight line), so walks up to that length are allowed.")
        log.info(walk_note)
        max_walk = need + 1.0
        s = Searcher(g, opts.cost, max_walk)
    best = s.max_bottleneck()
    log.info("max bottleneck: %.2f effective departures in window", best)

    routes = []
    rec = s.best_route(best)
    rec.label = "RECOMMENDED: least frequent leg as frequent as possible (then prefer local stops)"
    routes.append(rec)
    local = Searcher(g, CostParams.most_local(wait_factor=opts.cost.wait_factor, board_penalty_s=opts.cost.board_penalty_s,
                                              walk_factor=opts.cost.walk_factor), max_walk)
    for thr, label in ((best, "MOST LOCAL at the same weakest-link frequency"),
                       (float(local.levels[0]), "MOST LOCAL at any frequency")):
        r = local.best_route(thr)
        if r is not None and not any(r.nodes == x.nodes for x in routes):
            r.label = label
            routes.append(r)
    for k, r in enumerate(s.frontier(best)[1:], start=1):
        if not any(r.nodes == x.nodes for x in routes):
            r.label = f"TRADE-OFF {k}: weaker weakest link for a lower cost"
            routes.append(r)

    sims, profiles = [], []
    for r in routes:
        sim = Simulator(g, r)
        sims.append(sim)
        profiles.append(sim.profile())
    return Result(g, manifest, best, routes, profiles, sims, max_walk, walk_note)


def route_feeds(g: Graph, r: Route) -> set[str]:
    feeds = set()
    for leg in r.legs:
        if leg.kind == "ride":
            feeds |= {o.feed_id for o in serving_options(g, leg.u, leg.v)}
        elif leg.kind == "flex" and g.zones[leg.zone].source in g.feeds:
            feeds.add(g.zones[leg.zone].source)
    return feeds


def summarize_profile(profile) -> dict:
    if not profile:
        return {"connections": 0}
    durs = np.array([a - d for d, a in profile])
    return {
        "connections": len(profile),
        "min_duration_s": int(durs.min()),
        "median_duration_s": int(np.median(durs)),
        "max_duration_s": int(durs.max()),
    }


def gap_report(g: Graph, max_walk_m: float = np.inf, top: int = 8) -> str:
    """Where the network breaks: closest pairs of stops between the part
    reachable from the origin and the part that reaches the destination."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import breadth_first_order
    from scipy.spatial import cKDTree

    w = g.walk_dist <= max_walk_m
    u = np.concatenate([g.ride_u, g.flex_u, g.walk_u[w]])
    v = np.concatenate([g.ride_v, g.flex_v, g.walk_v[w]])
    m = csr_matrix((np.ones(len(u)), (u, v)), shape=(g.n, g.n))
    fw = breadth_first_order(m, g.origin, directed=True, return_predecessors=False)
    bw = breadth_first_order(m.T.tocsr(), g.destination, directed=True, return_predecessors=False)
    xy = project(g.nodes["lat"], g.nodes["lon"])
    d, j = cKDTree(xy[bw]).query(xy[fw])
    lines = ["Closest gaps between the part of the network reachable from South Station and the part that reaches Union Station:"]
    seen = set()
    for k in np.argsort(d):
        a, b = g.nodes.iloc[fw[k]], g.nodes.iloc[bw[j[k]]]
        key = (a["feed_id"], b["feed_id"])
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"  {d[k] / 1000:5.2f} km  {a['name']} [{a['feed_id']}] → {b['name']} [{b['feed_id']}]")
        if len(seen) >= top:
            break
    lines.append("Raise --gap-radius to bridge a gap on foot, or supply a Flex zone with --flex-zones.")
    return "\n".join(lines)
