"""Frequency graph construction.

Nodes are bus stops (identical stops published in several feeds, e.g. the MTA
borough feeds, are merged). Edges are:

* ride edges u -> v for every ordered stop pair (u before v) on some trip
  pattern where boarding at u and alighting at v are allowed, weighted by the
  *effective frequency* of departures from u (see `effective_frequency`).
  All patterns serving the same (u, v) pair are pooled by summing their
  frequencies. This solves the *common lines* problem — two hourly routes
  sharing a street give a 30-minute leg — and means riding through on one
  bus is always a single edge, so "stay on the bus" is never penalised as a
  transfer.
* walk edges between nearby stops (all pairs within `walk_radius_m`) plus
  "gap" links from each stop to the nearest few stops of every *other* feed
  within `gap_radius_m`, which is what stitches agencies together.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .flex import FlexZone, contains, nearest_boundary
from .geo import project, unproject
from .gtfs import CompiledFeed

log = logging.getLogger(__name__)

# Demand-response services restricted to eligible (ADA) riders.
PARATRANSIT_RE = re.compile(r"paratransit|\bada\b|access-?a-?ride|mobility\s*link", re.IGNORECASE)


def effective_frequency(dep: np.ndarray, w0: int, w1: int) -> np.ndarray:
    """Effective departures per window, per column of `dep` (trips x stops).

    With departures splitting the window into gaps g_i (the gap after the
    last departure wraps around to the first one, i.e. service outside the
    window is ignored), a rider turning up at a random moment waits
    E[w] = sum(g_i^2) / (2 W) on average. The *effective headway* is 2 E[w]
    and the effective frequency W / effective headway = W^2 / sum(g_i^2).

    For evenly spaced service this is just the number of departures. For
    bunched service it is much lower: four peak-only trips count as ~1.2, a
    single daily trip counts as 1. This is what makes a peak-only express no
    better than a daily bus when it is the only link in a chain.
    """
    dep = np.atleast_2d(np.asarray(dep, dtype=float))
    W = float(w1 - w0)
    x = np.where((dep >= w0) & (dep < w1), dep, np.inf)
    x.sort(axis=0)
    valid = np.isfinite(x)
    m = valid.sum(axis=0)
    both = valid[1:] & valid[:-1]
    gaps = np.where(both, np.diff(np.where(valid, x, 0.0), axis=0), 0.0)
    sumsq = (gaps ** 2).sum(axis=0)
    cols = np.arange(x.shape[1])
    first = np.where(m > 0, x[0], w0)
    last = np.where(m > 0, x[np.maximum(m - 1, 0), cols], w0)
    sumsq = sumsq + ((w1 - last) + (first - w0)) ** 2
    return np.where(m > 0, W * W / np.maximum(sumsq, 1.0), 0.0)


@dataclass
class GraphParams:
    window_start: int = 6 * 3600
    window_end: int = 22 * 3600
    walk_radius_m: float = 400.0
    gap_radius_m: float = 2500.0
    gap_neighbors: int = 3
    walk_speed_mps: float = 1.25
    walk_detour: float = 1.3
    access_radius_m: float = 800.0
    # Flex (demand-response) zones: equivalent headway used for frequency,
    # road speed and detour for in-vehicle time.
    flex_headway_s: int = 3600
    flex_speed_mps: float = 35 / 3.6
    flex_detour: float = 1.35

    @property
    def window_s(self) -> int:
        return self.window_end - self.window_start


@dataclass
class Graph:
    nodes: pd.DataFrame  # node -> stop_id, name, lat, lon, feed_id
    # ride edges (pooled over patterns)
    ride_u: np.ndarray
    ride_v: np.ndarray
    ride_freq: np.ndarray  # effective departures per window serving u -> v
    ride_time: np.ndarray  # seconds, frequency-weighted mean in-vehicle time
    # walk edges
    walk_u: np.ndarray
    walk_v: np.ndarray
    walk_time: np.ndarray
    params: GraphParams
    feeds: dict[str, CompiledFeed]
    # (feed_id, pattern index) and node ids of each pattern's stops
    pattern_nodes: dict[tuple[str, int], np.ndarray]
    origin: int = -1
    destination: int = -1
    # flex edges (one per ordered access-point pair of a zone)
    flex_u: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    flex_v: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    flex_freq: np.ndarray = field(default_factory=lambda: np.zeros(0))
    flex_time: np.ndarray = field(default_factory=lambda: np.zeros(0))
    flex_zone: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    zones: list = field(default_factory=list)
    day: object = None  # representative date (for Flex service hours)

    @property
    def n(self) -> int:
        return len(self.nodes)


def _merge_stops(feeds: list[CompiledFeed]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    frames = []
    for f in feeds:
        s = f.stops.copy()
        s["feed_id"] = f.feed_id
        s["local"] = np.arange(len(s))
        frames.append(s)
    allstops = pd.concat(frames, ignore_index=True)
    # Same stop_id at (practically) the same spot in two feeds = same stop.
    key = allstops["stop_id"] + "|" + allstops["lat"].round(4).astype(str) + "|" + allstops["lon"].round(4).astype(str)
    node_of_row, uniques = pd.factorize(key)
    first_row = pd.Series(np.arange(len(allstops))).groupby(node_of_row).first().to_numpy()
    nodes = allstops.iloc[first_row][["stop_id", "name", "lat", "lon", "feed_id"]].reset_index(drop=True)
    mapping: dict[str, np.ndarray] = {}
    for f in feeds:
        rows = np.flatnonzero((allstops["feed_id"] == f.feed_id).to_numpy())
        m = np.empty(len(f.stops), dtype=np.int64)
        m[allstops["local"].to_numpy()[rows]] = node_of_row[rows]
        mapping[f.feed_id] = m
    return nodes, mapping


def _pattern_pairs(nd: np.ndarray, board, alight, dep, arr, p: GraphParams):
    L = len(nd)
    freq = effective_frequency(dep, p.window_start, p.window_end)
    if not freq.any():
        return None
    # Typical running-time profile relative to the trip start, from the trips
    # that run inside the window.
    in_win = (dep >= p.window_start) & (dep < p.window_end)
    rows = in_win.any(axis=1)
    base = dep[rows, :1]
    prof_dep = np.median(dep[rows] - base, axis=0)
    prof_arr = np.median(arr[rows] - base, axis=0)
    I, J = np.triu_indices(L, 1)
    ok = board[I] & alight[J] & (freq[I] > 0) & (nd[I] != nd[J])
    I, J = I[ok], J[ok]
    ride = np.maximum(prof_arr[J] - prof_dep[I], 30.0)
    return nd[I], nd[J], freq[I], ride


def _aggregate(u, v, freq, time, n):
    """Pool parallel (u, v) ride edges: sum frequencies, frequency-weighted mean time."""
    key = u.astype(np.int64) * n + v
    uniq, inv = np.unique(key, return_inverse=True)
    f = np.bincount(inv, weights=freq)
    t = np.bincount(inv, weights=time * freq) / f
    return (uniq // n).astype(np.int64), (uniq % n).astype(np.int64), f, t


def _merged_patterns(feeds: list[CompiledFeed], mapping):
    """Patterns keyed by their node sequence. The same trips published in two
    feeds (agency mirrors, merged operators) collapse into one pattern with
    the union of their departures instead of being counted twice."""
    merged: dict[bytes, list] = {}
    for f in feeds:
        for pat in f.patterns:
            nd = mapping[f.feed_id][pat.stops]
            key = nd.tobytes() + pat.board.tobytes() + pat.alight.tobytes()
            if key in merged:
                m = merged[key]
                dep = np.vstack([m[3], pat.dep])
                arr = np.vstack([m[4], pat.arr])
                _, keep = np.unique(dep[:, 0], return_index=True)
                m[3], m[4] = dep[keep], arr[keep]
            else:
                merged[key] = [nd, pat.board, pat.alight, pat.dep, pat.arr, f.feed_id]
    return merged


def build_graph(feeds: list[CompiledFeed], params: GraphParams) -> Graph:
    nodes, mapping = _merge_stops(feeds)
    n = len(nodes)
    log.info("graph: %d stops from %d feeds", n, len(feeds))
    pattern_nodes = {(f.feed_id, k): mapping[f.feed_id][pat.stops] for f in feeds for k, pat in enumerate(f.patterns)}

    by_feed: dict[str, list] = {}
    for nd, board, alight, dep, arr, feed_id in _merged_patterns(feeds, mapping).values():
        res = _pattern_pairs(nd, board, alight, dep.astype(np.int64), arr.astype(np.int64), params)
        if res is not None:
            by_feed.setdefault(feed_id, []).append(res)
    parts = []
    for chunks in by_feed.values():
        # Pre-aggregate per feed to bound peak memory.
        parts.append(_aggregate(*(np.concatenate([c[i] for c in chunks]) for i in range(4)), n))
    ru, rv, rf, rt = _aggregate(*(np.concatenate([p[i] for p in parts]) for i in range(4)), n)
    log.info("graph: %d pooled ride edges", len(ru))

    wu, wv, wt = _walk_edges(nodes, params)
    log.info("graph: %d walk edges", len(wu))
    return Graph(nodes, ru, rv, rf, rt, wu, wv, wt, params, {f.feed_id: f for f in feeds}, pattern_nodes)


def _walk_seconds(dist_m, p: GraphParams):
    return np.maximum(dist_m * p.walk_detour / p.walk_speed_mps, 1.0)


def _walk_edges(nodes: pd.DataFrame, p: GraphParams):
    xy = project(nodes["lat"], nodes["lon"])
    tree = cKDTree(xy)
    pairs = tree.query_pairs(p.walk_radius_m, output_type="ndarray")
    us, vs = [pairs[:, 0], pairs[:, 1]], [pairs[:, 1], pairs[:, 0]]

    # Inter-agency gap links: nearest stops of each other feed.
    feed_codes, feed_names = pd.factorize(nodes["feed_id"])
    for code in range(len(feed_names)):
        own = np.flatnonzero(feed_codes == code)
        others = np.flatnonzero(feed_codes != code)
        lo, hi = xy[own].min(axis=0) - p.gap_radius_m, xy[own].max(axis=0) + p.gap_radius_m
        near = others[np.all((xy[others] >= lo) & (xy[others] <= hi), axis=1)]
        if not len(near):
            continue
        ftree = cKDTree(xy[own])
        k = min(p.gap_neighbors, len(own))
        d, j = ftree.query(xy[near], k=k, distance_upper_bound=p.gap_radius_m)
        d, j = d.reshape(len(near), k), j.reshape(len(near), k)
        ok = np.isfinite(d) & (d > p.walk_radius_m)
        src = np.repeat(near, k).reshape(len(near), k)[ok]
        dst = own[j[ok]]
        us += [src, dst]
        vs += [dst, src]
    u = np.concatenate(us).astype(np.int64)
    v = np.concatenate(vs).astype(np.int64)
    key = np.unique(u * len(nodes) + v)
    u, v = key // len(nodes), key % len(nodes)
    dist = np.linalg.norm(xy[u] - xy[v], axis=1)
    return u, v, _walk_seconds(dist, p)


def attach_places(g: Graph, origin, destination) -> Graph:
    """Add origin/destination nodes linked by walking to stops near each."""
    p = g.params
    xy = project(g.nodes["lat"], g.nodes["lon"])
    tree = cKDTree(xy)
    o, d = g.n, g.n + 1
    new_nodes = pd.DataFrame([
        {"stop_id": "ORIGIN", "name": origin.name, "lat": origin.lat, "lon": origin.lon, "feed_id": ""},
        {"stop_id": "DESTINATION", "name": destination.name, "lat": destination.lat, "lon": destination.lon, "feed_id": ""},
    ])
    wu, wv, wt = [g.walk_u], [g.walk_v], [g.walk_time]
    for place, node, outbound in ((origin, o, True), (destination, d, False)):
        pxy = project([place.lat], [place.lon])[0]
        near = np.array(tree.query_ball_point(pxy, p.access_radius_m), dtype=np.int64)
        if not len(near):
            raise RuntimeError(f"no bus stop within {p.access_radius_m:.0f} m of {place.name}")
        t = _walk_seconds(np.linalg.norm(xy[near] - pxy, axis=1), p)
        if outbound:
            wu.append(np.full(len(near), node)); wv.append(near)
        else:
            wu.append(near); wv.append(np.full(len(near), node))
        wt.append(t)
    g.nodes = pd.concat([g.nodes, new_nodes], ignore_index=True)
    g.walk_u, g.walk_v, g.walk_time = np.concatenate(wu), np.concatenate(wv), np.concatenate(wt)
    g.origin, g.destination = o, d
    return g


def flex_departures(zone: FlexZone, day, p: GraphParams) -> np.ndarray:
    """Synthetic departures standing in for on-demand service: one per
    equivalent headway (or the zone's own published frequency) while the
    zone is running."""
    deps = [np.arange(s0, s1 + 1, headway or p.flex_headway_s) for s0, s1, headway in zone.windows(day)]
    return np.concatenate(deps) if deps else np.zeros(0)


def flex_frequency(zone: FlexZone, day, p: GraphParams) -> float:
    d = flex_departures(zone, day, p)
    return float(effective_frequency(d[:, None], p.window_start, p.window_end)[0]) if len(d) else 0.0


def add_flex(g: Graph, zones: list[FlexZone], day, max_pairwise: int = 700) -> Graph:
    """Wire demand-response zones into the graph (see flex.py)."""
    p = g.params
    xy = project(g.nodes["lat"], g.nodes["lon"])
    lat, lon = g.nodes["lat"].to_numpy(), g.nodes["lon"].to_numpy()
    new_nodes, fu, fv, fc, ft, fz = [], [], [], [], [], []
    wu, wv, wt = [g.walk_u], [g.walk_v], [g.walk_time]
    next_id = g.n
    kept: list[FlexZone] = []
    for zone in zones:
        if PARATRANSIT_RE.search(f"{zone.name} {zone.agency}"):
            log.info("skip %s: paratransit is limited to eligible riders", zone.name)
            continue
        freq = flex_frequency(zone, day, p)
        if freq == 0:
            continue
        la0, la1, lo0, lo1 = zone.bbox()
        pad_lat = p.gap_radius_m / 111_000
        pad_lon = pad_lat / np.cos(np.radians((la0 + la1) / 2))
        cand = np.flatnonzero((lat >= la0 - pad_lat) & (lat <= la1 + pad_lat) & (lon >= lo0 - pad_lon) & (lon <= lo1 + pad_lon))
        if not len(cand):
            continue
        inside = contains(zone, lat[cand], lon[cand])
        access = list(cand[inside])
        access_xy = [xy[cand[inside]]]
        outside = cand[~inside]
        if len(outside):
            d, q = nearest_boundary(zone, lat[outside], lon[outside])
            near = d <= p.gap_radius_m
            # One zone-edge point per ~250 m of boundary, shared by the
            # outside stops that map onto it.
            cells: dict[tuple[int, int], int] = {}
            for stop, dist, qxy in zip(outside[near], d[near], q[near]):
                key = (int(qxy[0] // 250), int(qxy[1] // 250))
                if key not in cells:
                    cells[key] = next_id
                    qlat, qlon = unproject(qxy)
                    new_nodes.append({"stop_id": f"FLEXEDGE:{zone.zone_id}:{len(cells)}", "name": f"{zone.name} zone edge",
                                      "lat": float(qlat[0]), "lon": float(qlon[0]), "feed_id": f"flex:{zone.zone_id}"})
                    access.append(next_id)
                    access_xy.append(qxy[None, :])
                    next_id += 1
                node = cells[key]
                t = _walk_seconds(np.array([dist]), p)
                wu += [np.array([stop, node])]
                wv += [np.array([node, stop])]
                wt += [np.repeat(t, 2)]
        access = np.array(access, dtype=np.int64)
        axy = np.vstack(access_xy)
        if len(access) < 2:
            continue
        if len(access) > max_pairwise:
            log.warning("flex zone %s has %d access points; keeping the %d nearest to its centroid", zone.name, len(access), max_pairwise)
            keep = np.argsort(np.linalg.norm(axy - axy.mean(0), axis=1))[:max_pairwise]
            access, axy = access[keep], axy[keep]
        I, J = np.nonzero(~np.eye(len(access), dtype=bool))
        dist = np.linalg.norm(axy[I] - axy[J], axis=1)
        fu.append(access[I]); fv.append(access[J])
        fc.append(np.full(len(I), freq)); ft.append(np.maximum(dist * p.flex_detour / p.flex_speed_mps, 60.0))
        fz.append(np.full(len(I), len(kept)))
        kept.append(zone)
        log.info("flex zone %s: %d access points, effective frequency %.1f", zone.name, len(access), freq)
    if new_nodes:
        g.nodes = pd.concat([g.nodes, pd.DataFrame(new_nodes)], ignore_index=True)
    g.walk_u, g.walk_v, g.walk_time = np.concatenate(wu), np.concatenate(wv), np.concatenate(wt)
    if fu:
        g.flex_u, g.flex_v = np.concatenate(fu), np.concatenate(fv)
        g.flex_freq, g.flex_time, g.flex_zone = np.concatenate(fc), np.concatenate(ft), np.concatenate(fz)
    g.zones = kept
    g.day = day
    return g
