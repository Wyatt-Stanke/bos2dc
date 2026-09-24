"""Road-based locality: how local is the road a bus is driving on?

Classes, from most to least local:

    0  1 lane   one through lane in the bus's direction
    1  2 lanes  two lanes in the bus's direction
    2  3+ lanes three or more
    3  controlled access  motorways and their ramps, and roads tagged
                          motorroad=yes (no at-grade access)

Lane counts come from OpenStreetMap. `lanes` on a one-way way (including each
carriageway of a divided road) is the count in the direction of travel. On a
two-way way `lanes:forward` / `lanes:backward` are used when present, and
otherwise half of `lanes`, rounded down, so `lanes=3` (a centre turn lane) and
`lanes=2` are both one lane each way. Untagged ways take the value of the
nearest tagged way of the same street (same name or ref, highway type and
one-way status, within 1.5 km). Failing that, they take the length-weighted
median of their highway type and one-way status in the extract.

Buses are placed on roads by matching their GTFS shapes. Each shape is
resampled every 20 m, and each piece is matched to the nearest road segment
within 25 m whose direction is within 40 degrees of the shape's. Heading is
what keeps a bus crossing a freeway on an overpass off the freeway, and what
picks the right carriageway. Tunnels and bridges only win when no street at
ground level is about as close, so a freeway tunnel under a street does not
capture the buses on the street. Short blips (turn-lane pockets, ramps
brushed at junctions) are smoothed out, and unmatched pieces take their
neighbours' class. Stops are aligned to the shape in order, by dynamic
programming, so each hop between stops gets miles per class.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
import shutil
import subprocess
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .geo import REF_LAT, project

log = logging.getLogger(__name__)

ROAD_CLASS_NAMES = ("1 lane", "2 lanes", "3+ lanes", "controlled access")
ONE_LANE, TWO_LANES, THREE_PLUS, CONTROLLED = 0, 1, 2, 3
UNMATCHED = -1

# highway=* values a bus can run on. Driveways and drive-throughs are left
# out (millions of ways, never on a route).
BUS_HIGHWAYS = (
    "motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link", "secondary", "secondary_link",
    "tertiary", "tertiary_link", "unclassified", "residential", "living_street", "service", "busway", "bus_guideway", "road",
)
SKIP_SERVICE = {"driveway", "drive-through", "emergency_access"}

# Download sources for the corridor extracts, tried in order. Geofabrik is
# the canonical host; openstreetmap.fr mirrors the same regional extracts.
EXTRACTS = ("us-northeast", "us-south")
MIRRORS = (
    "https://download.geofabrik.de/north-america/{name}-latest.osm.pbf",
    "https://download.openstreetmap.fr/extracts/north-america/{name}.osm.pbf",
)

ROADS_VERSION = 2
PIECE_M = 20.0
MATCH_RADIUS_M = 25.0
MAX_HEADING_DEG = 40.0
HEADING_WEIGHT_M_PER_DEG = 0.4  # 10 degrees of misalignment "costs" 4 m of distance
CONTRAFLOW_PENALTY_M = 15.0  # riding a one-way way backwards (contraflow bus lane)
# Tunnels, bridges and anything else off the ground layer: a freeway tunnel
# under a street (Boston's Central Artery) or a viaduct over one must not
# capture the buses running on the street.
OFF_GRADE_PENALTY_M = 12.0
MIN_RUN_M = 150.0
MAX_STOP_OFFSET_M = 150.0  # median stop-to-shape distance beyond which a shape is not trusted
INDEX_PIECE_M = 40.0


# --------------------------------------------------------------------------- lanes


def _num(v: str | None) -> float:
    if not v:
        return np.nan
    try:
        return float(v.split(";")[0].strip())
    except ValueError:
        return np.nan


def lanes_by_direction(tags: dict) -> tuple[int, float, float]:
    """(oneway, lanes forward, lanes backward) from a way's tags.

    oneway is 1 (with the way), -1 (against it) or 0. For a one-way way the
    opposite direction is a contraflow lane: lanes:backward if tagged, else
    one lane."""
    hw = tags.get("highway", "")
    ow = tags.get("oneway", "")
    if ow in ("yes", "1", "true") or tags.get("junction") in ("roundabout", "circular") or (hw in ("motorway", "motorway_link") and ow != "no"):
        oneway = 1
    elif ow == "-1":
        oneway = -1
    else:
        oneway = 0
    lanes = _num(tags.get("lanes"))
    fwd, bwd = _num(tags.get("lanes:forward")), _num(tags.get("lanes:backward"))
    if oneway == 1:
        return 1, (lanes if np.isfinite(lanes) else fwd), (bwd if np.isfinite(bwd) else 1.0)
    if oneway == -1:
        return -1, (fwd if np.isfinite(fwd) else 1.0), (lanes if np.isfinite(lanes) else bwd)
    if np.isfinite(fwd) or np.isfinite(bwd):
        per = np.nanmax([fwd, bwd])
        return 0, (fwd if np.isfinite(fwd) else per), (bwd if np.isfinite(bwd) else per)
    if np.isfinite(lanes):
        per = max(1.0, np.floor(lanes / 2))
        return 0, per, per
    return 0, np.nan, np.nan


def lane_class(lanes: np.ndarray) -> np.ndarray:
    return np.where(lanes < 1.5, ONE_LANE, np.where(lanes < 2.5, TWO_LANES, THREE_PLUS)).astype(np.int8)


def is_controlled(tags: dict) -> bool:
    hw = tags.get("highway", "")
    return hw in ("motorway", "motorway_link") or tags.get("motorroad") == "yes"


def off_grade(tags) -> bool:
    """Tunnel, bridge, or a non-zero layer."""
    tunnel, bridge, layer = tags.get("tunnel"), tags.get("bridge"), tags.get("layer")
    return (tunnel not in (None, "no")) or (bridge not in (None, "no")) or (layer not in (None, "0"))


# --------------------------------------------------------------------------- building the index


def ensure_extracts(osm_dir: Path) -> list[Path]:
    """Download the regional extracts (resumable) unless already present."""
    import requests

    osm_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for name in EXTRACTS:
        dest = osm_dir / f"{name}.osm.pbf"
        out.append(dest)
        if dest.exists() and not (osm_dir / f"{name}.osm.pbf.part").exists():
            continue
        tmp = osm_dir / f"{name}.osm.pbf.part"
        src = osm_dir / f"{name}.osm.pbf.src"  # which mirror the partial file came from
        last = None
        for tmpl in MIRRORS:
            url = tmpl.format(name=name)
            try:
                if tmp.exists() and (not src.exists() or src.read_text() != url):
                    tmp.unlink()  # mirrors publish different builds: never splice them
                src.write_text(url)
                have = tmp.stat().st_size if tmp.exists() else 0
                headers = {"Range": f"bytes={have}-"} if have else {}
                with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                    if r.status_code == 416:
                        break
                    r.raise_for_status()
                    mode = "ab" if have and r.status_code == 206 else "wb"
                    log.info("downloading %s", url)
                    with open(tmp, mode) as fh:
                        for chunk in r.iter_content(1 << 22):
                            fh.write(chunk)
                break
            except Exception as e:  # noqa: BLE001 - try the next mirror
                last = e
                log.warning("download %s failed: %s", url, e)
        else:
            raise RuntimeError(f"could not download {name}: {last}")
        tmp.rename(dest)
        src.unlink(missing_ok=True)
    return out


def cut_to_corridor(pbfs: list[Path], bbox: tuple[float, float, float, float], osm_dir: Path) -> list[Path]:
    """osmium-tool: clip each extract to the bbox and keep bus-usable road ways."""
    if not shutil.which("osmium"):
        raise RuntimeError("osmium-tool is required to cut the OpenStreetMap extracts (apt install osmium-tool)")
    lon0, lat0, lon1, lat1 = bbox
    out = []
    for pbf in pbfs:
        clip = osm_dir / f"{pbf.name.split('.')[0]}-corridor.osm.pbf"
        roads = osm_dir / f"{pbf.name.split('.')[0]}-roads.osm.pbf"
        if not roads.exists() or roads.stat().st_mtime < pbf.stat().st_mtime:
            subprocess.run(["osmium", "extract", "-b", f"{lon0},{lat0},{lon1},{lat1}", "--strategy", "complete_ways",
                            str(pbf), "-o", str(clip), "--overwrite"], check=True)
            subprocess.run(["osmium", "tags-filter", str(clip), "w/highway=" + ",".join(BUS_HIGHWAYS), "-o", str(roads), "--overwrite"],
                           check=True)
            clip.unlink()
        out.append(roads)
    return out


def _read_ways(pbf: Path):
    from array import array

    import osmium

    hw_code = {h: i for i, h in enumerate(BUS_HIGHWAYS)}
    ids, hw, oneway, fwd, bwd, ca, grade, name, counts = (array("q"), array("b"), array("b"), array("d"), array("d"), array("b"),
                                                        array("b"), [], array("q"))
    xs, ys = array("d"), array("d")
    fp = osmium.FileProcessor(str(pbf)).with_locations().with_filter(osmium.filter.EntityFilter(osmium.osm.WAY))
    for w in fp:
        t = w.tags
        h = t.get("highway")
        if h not in hw_code or t.get("service") in SKIP_SERVICE or t.get("area") == "yes":
            continue
        n0 = len(xs)
        for n in w.nodes:
            if n.location.valid():
                xs.append(n.lon)
                ys.append(n.lat)
        if len(xs) - n0 < 2:
            del xs[n0:], ys[n0:]
            continue
        tags = {k: t.get(k) for k in ("highway", "oneway", "junction", "lanes", "lanes:forward", "lanes:backward", "motorroad") if k in t}
        o, f, b = lanes_by_direction(tags)
        ids.append(w.id)
        hw.append(hw_code[h])
        oneway.append(o)
        fwd.append(f)
        bwd.append(b)
        ca.append(is_controlled(tags))
        grade.append(off_grade(t))
        name.append(t.get("name") or t.get("ref") or "")
        counts.append(len(xs) - n0)
    return dict(id=np.frombuffer(ids, np.int64), hw=np.frombuffer(hw, np.int8), oneway=np.frombuffer(oneway, np.int8),
                fwd=np.frombuffer(fwd, float), bwd=np.frombuffer(bwd, float), ca=np.frombuffer(ca, np.int8).astype(bool),
                off_grade=np.frombuffer(grade, np.int8).astype(bool),
                name=np.array(name, object), lon=np.frombuffer(xs, float), lat=np.frombuffer(ys, float),
                counts=np.frombuffer(counts, np.int64))


def _impute(w: dict, xy: np.ndarray, offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fill missing lanes per direction. Returns fwd, bwd and the source of
    the value (0 tagged, 1 same street nearby, 2 highway-type median)."""
    fwd, bwd = w["fwd"].copy(), w["bwd"].copy()
    missing = ~np.isfinite(fwd) | ~np.isfinite(bwd)
    source = np.where(missing, 2, 0).astype(np.int8)
    both_known = np.isfinite(fwd) & np.isfinite(bwd)
    per_way = np.where(w["oneway"] == 1, fwd, np.where(w["oneway"] == -1, bwd, np.where(both_known, (fwd + bwd) / 2, np.nan)))
    starts, ends = offsets[:-1], offsets[1:]
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    csum = np.r_[0.0, np.cumsum(seg)]
    length = csum[ends - 1] - csum[starts]
    centroid = np.column_stack([np.add.reduceat(xy[:, 0], starts) / (ends - starts), np.add.reduceat(xy[:, 1], starts) / (ends - starts)])
    base_hw = np.array([BUS_HIGHWAYS.index(h.replace("_link", "")) if h.endswith("_link") else i for i, h in enumerate(BUS_HIGHWAYS)])[w["hw"]]
    ow = (w["oneway"] != 0).astype(np.int8)

    # 1. Same street: nearest tagged way with the same name/ref, road type and one-way status.
    has = np.isfinite(per_way)
    named = w["name"] != ""
    key = pd.Series(w["name"]).astype(str) + "|" + pd.Series(base_hw).astype(str) + "|" + pd.Series(ow).astype(str)
    codes, _ = pd.factorize(key)
    order = np.argsort(codes, kind="stable")
    bounds = np.flatnonzero(np.r_[True, codes[order][1:] != codes[order][:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        members = order[a:b]
        donors = members[has[members]]
        takers = members[missing[members] & named[members]]
        if not len(donors) or not len(takers):
            continue
        d, j = cKDTree(centroid[donors]).query(centroid[takers], distance_upper_bound=1500.0)
        ok = np.isfinite(d)
        val = per_way[donors[j[ok]]]
        t = takers[ok]
        fwd[t] = np.where(np.isfinite(w["fwd"][t]), w["fwd"][t], val)
        bwd[t] = np.where(np.isfinite(w["bwd"][t]), w["bwd"][t], np.where(w["oneway"][t] != 0, 1.0, val))
        source[t] = 1
    # 2. Highway type median (length-weighted) for what is left.
    still = ~np.isfinite(fwd) | ~np.isfinite(bwd)
    for h in np.unique(base_hw[still]):
        for o in (0, 1):
            grp = (base_hw == h) & (ow == o)
            donors = grp & has
            if donors.any():
                srt = np.argsort(per_way[donors])
                cw = np.cumsum(length[donors][srt])
                med = float(per_way[donors][srt][np.searchsorted(cw, cw[-1] / 2)])
            else:
                med = 1.0
            med = max(1.0, round(med))
            t = grp & still
            fwd[t] = np.where(np.isfinite(fwd[t]), fwd[t], med)
            bwd[t] = np.where(np.isfinite(bwd[t]), bwd[t], 1.0 if o else med)
    return fwd, bwd, source


def build_index(pbfs: list[Path], out: Path) -> Path:
    """Road ways of the corridor as node coordinates plus per-way lane
    classes (the matcher splits and prunes segments at load time)."""
    parts = [_read_ways(p) for p in pbfs]
    w = {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k not in ("lon", "lat", "counts")}
    counts = np.concatenate([p["counts"] for p in parts])
    lon = np.concatenate([p["lon"] for p in parts])
    lat = np.concatenate([p["lat"] for p in parts])
    del parts
    # Ways straddling two extracts appear in both; keep the first copy.
    _, first = np.unique(w["id"], return_index=True)
    if len(first) < len(counts):
        keep_way = np.zeros(len(counts), bool)
        keep_way[first] = True
        node_keep = np.repeat(keep_way, counts)
        w = {k: v[keep_way] for k, v in w.items()}
        counts = counts[keep_way]
        lon, lat = lon[node_keep], lat[node_keep]
    offsets = np.r_[0, np.cumsum(counts)]
    xy = project(lat, lon)
    del lon, lat
    fwd, bwd, source = _impute(w, xy, offsets)
    cls_f = np.where(w["ca"], CONTROLLED, lane_class(fwd)).astype(np.int8)
    cls_b = np.where(w["ca"], CONTROLLED, lane_class(bwd)).astype(np.int8)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, xy=xy, offsets=offsets, cls_f=cls_f, cls_b=cls_b, oneway=w["oneway"], source=source, hw=w["hw"], way_id=w["id"],
             off_grade=w["off_grade"])
    tmp.rename(out)
    log.info("road index: %d ways, %d nodes; lanes tagged on %.0f%% of ways, same-street %.0f%%, type median %.0f%%",
             len(counts), len(xy), *(100 * np.mean(source == s) for s in (0, 1, 2)))
    return out


CELL_M = 50.0


def _cells_of(xy: np.ndarray) -> np.ndarray:
    ij = np.floor(xy / CELL_M).astype(np.int64) + (1 << 20)
    return ij[:, 0] * (1 << 21) + ij[:, 1]


def cell_keys(xy: np.ndarray, dilate: int = 1) -> np.ndarray:
    """Grid cells (CELL_M) covering the points, grown by `dilate` cells."""
    base = np.unique(_cells_of(xy))
    i, j = base // (1 << 21), base % (1 << 21)
    out = [(i + di) * (1 << 21) + (j + dj) for di in range(-dilate, dilate + 1) for dj in range(-dilate, dilate + 1)]
    return np.unique(np.concatenate(out))


@dataclass
class RoadIndex:
    """Road pieces (<= INDEX_PIECE_M) near the bus shapes, with a KD-tree on their midpoints."""
    p0: np.ndarray
    p1: np.ndarray
    way: np.ndarray
    cls_f: np.ndarray
    cls_b: np.ndarray
    oneway: np.ndarray
    source: np.ndarray
    off_grade: np.ndarray
    tree: cKDTree

    @staticmethod
    def digest(path: Path) -> str:
        st = path.stat()
        return hashlib.sha1(f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}|{ROADS_VERSION}".encode()).hexdigest()[:12]

    @classmethod
    def load(cls, path: Path, near: np.ndarray | None = None, chunk: int = 4_000_000) -> "RoadIndex":
        """Load the index, keeping only pieces whose midpoint lies in one of
        the grid cells `near` (see cell_keys)."""
        z = np.load(path)
        xy, offsets = z["xy"], z["offsets"]
        way_of_node = np.repeat(np.arange(len(offsets) - 1, dtype=np.int32), np.diff(offsets))
        seg = np.flatnonzero(way_of_node[1:] == way_of_node[:-1])
        p0s, p1s, ways = [], [], []
        for c in range(0, len(seg), chunk):
            s = seg[c:c + chunk]
            a, b = xy[s], xy[s + 1]
            L = np.linalg.norm(b - a, axis=1)
            k = np.maximum(1, np.ceil(L / INDEX_PIECE_M)).astype(np.int64)
            rep = np.repeat(np.arange(len(s)), k)
            step = np.arange(len(rep)) - np.repeat(np.cumsum(k) - k, k)
            t0, t1 = step / k[rep], (step + 1) / k[rep]
            d = b[rep] - a[rep]
            q0 = a[rep] + t0[:, None] * d
            q1 = a[rep] + t1[:, None] * d
            keep = np.linalg.norm(q1 - q0, axis=1) > 0.01
            if near is not None:
                keep &= np.isin(_cells_of((q0 + q1) / 2), near)
            p0s.append(q0[keep]); p1s.append(q1[keep]); ways.append(way_of_node[s][rep][keep])
        p0, p1 = np.concatenate(p0s), np.concatenate(p1s)
        log.info("road index: %d pieces kept", len(p0))
        grade = z["off_grade"] if "off_grade" in z.files else np.zeros(len(z["cls_f"]), bool)
        return cls(p0, p1, np.concatenate(ways), z["cls_f"], z["cls_b"], z["oneway"], z["source"], grade, cKDTree((p0 + p1) / 2))

    def match(self, xy: np.ndarray, heading: np.ndarray, chunk: int = 250_000, k: int = 24) -> tuple[np.ndarray, np.ndarray]:
        """Road class (or UNMATCHED) and lanes source for each point."""
        out_c = np.full(len(xy), UNMATCHED, np.int8)
        out_s = np.full(len(xy), -1, np.int8)
        r = MATCH_RADIUS_M + INDEX_PIECE_M / 2
        cos_max = np.cos(np.radians(MAX_HEADING_DEG))
        n_pieces = len(self.p0)
        for s in range(0, len(xy), chunk):
            q = xy[s:s + chunk]
            h = heading[s:s + chunk]
            _, jj = self.tree.query(q, k=k, distance_upper_bound=r, workers=-1)
            qi, col = np.nonzero(jj < n_pieces)
            if not len(qi):
                continue
            si = jj[qi, col]
            a, b = self.p0[si], self.p1[si]
            d = b - a
            L2 = np.einsum("ij,ij->i", d, d)
            t = np.clip(np.einsum("ij,ij->i", q[qi] - a, d) / L2, 0, 1)
            dist = np.linalg.norm(q[qi] - (a + t[:, None] * d), axis=1)
            cosang = np.einsum("ij,ij->i", d, h[qi]) / np.sqrt(L2)
            forward = cosang >= 0
            acos = np.minimum(np.abs(cosang), 1.0)
            wy = self.way[si]
            ow = self.oneway[wy]
            against = ((ow == 1) & ~forward) | ((ow == -1) & forward)
            score = (dist + HEADING_WEIGHT_M_PER_DEG * np.degrees(np.arccos(acos)) + CONTRAFLOW_PENALTY_M * against
                     + OFF_GRADE_PENALTY_M * self.off_grade[wy])
            score = np.where((dist <= MATCH_RADIUS_M) & (acos >= cos_max), score, np.inf)
            order = np.lexsort((score, qi))
            qs = qi[order]
            best = order[np.r_[True, qs[1:] != qs[:-1]]]
            best = best[np.isfinite(score[best])]
            wb = wy[best]
            out_c[s + qi[best]] = np.where(forward[best], self.cls_f[wb], self.cls_b[wb])
            out_s[s + qi[best]] = self.source[wb]
        return out_c, out_s


def _smooth(cls: np.ndarray, piece_len: np.ndarray) -> np.ndarray:
    """Fill unmatched pieces from their neighbours and remove short blips."""
    c = cls.copy()
    if (c == UNMATCHED).all():
        return c
    # Unmatched: take the previous matched class (forward fill), then backward fill the head.
    idx = np.where(c != UNMATCHED, np.arange(len(c)), -1)
    np.maximum.accumulate(idx, out=idx)
    head = idx < 0
    c = np.where(head, UNMATCHED, c[np.maximum(idx, 0)])
    if head.any():
        c[head] = c[np.flatnonzero(~head)[0]]
    for _ in range(4):
        starts = np.flatnonzero(np.r_[True, c[1:] != c[:-1]])
        if len(starts) < 3:
            break
        ends = np.r_[starts[1:], len(c)]
        cs = np.r_[0.0, np.cumsum(piece_len)]
        run_len = cs[ends] - cs[starts]
        run_cls = c[starts]
        changed = False
        for k in np.argsort(run_len):
            if run_len[k] >= MIN_RUN_M:
                break
            if 0 < k < len(starts) - 1 and run_cls[k - 1] == run_cls[k + 1] and run_cls[k] != run_cls[k - 1]:
                c[starts[k]:ends[k]] = run_cls[k - 1]
                changed = True
        if not changed:
            break
    return c


@dataclass
class ShapeMatch:
    s: np.ndarray  # chainage (m) of the piece boundaries, length m + 1
    xy: np.ndarray  # boundary points, (m + 1, 2)
    cls: np.ndarray  # class per piece (UNMATCHED if nothing matched at all), length m
    source: np.ndarray  # lanes source per piece (-1 unmatched)
    length: np.ndarray  # true length (m) per piece


def resample(lat: np.ndarray, lon: np.ndarray, step: float = PIECE_M):
    xy = project(lat, lon)
    keep = np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 0.01]
    xy = xy[keep]
    if len(xy) < 2:
        return None
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    cs = np.r_[0.0, np.cumsum(seg)]
    total = cs[-1]
    m = max(1, int(np.ceil(total / step)))
    s = np.minimum(np.arange(m + 1) * step, total)
    s[-1] = total
    bx = np.interp(s, cs, xy[:, 0])
    by = np.interp(s, cs, xy[:, 1])
    pts = np.column_stack([bx, by])
    mid_s = (s[:-1] + s[1:]) / 2
    mid = np.column_stack([np.interp(mid_s, cs, xy[:, 0]), np.interp(mid_s, cs, xy[:, 1])])
    # Heading of the original polyline segment containing each midpoint.
    k = np.clip(np.searchsorted(cs, mid_s, side="right") - 1, 0, len(seg) - 1)
    hd = (xy[k + 1] - xy[k]) / seg[k][:, None]
    # True lengths: undo the projection's east-west scale at each latitude.
    lat_mid = np.degrees(mid[:, 1] / 6_371_000.0)
    kx = np.cos(np.radians(lat_mid)) / np.cos(np.radians(REF_LAT))
    dp = np.diff(pts, axis=0)
    length = np.hypot(dp[:, 0] * kx, dp[:, 1])
    return s, pts, mid, hd, length


def align_stops(stop_xy: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, float]:
    """Monotone assignment of stops (in order) to shape boundary points
    minimising the summed distance. Returns point indices and the median
    stop-to-shape distance."""
    n, m = len(stop_xy), len(pts)
    D = np.linalg.norm(stop_xy[:, None, :] - pts[None, :, :], axis=2)
    best = D[0].copy()
    back = np.empty((n, m), np.int32)
    idx = np.arange(m)
    for i in range(1, n):
        pm = np.minimum.accumulate(best)
        back[i] = np.maximum.accumulate(np.where(best == pm, idx, 0))
        best = D[i] + pm
    j = np.empty(n, np.int64)
    j[-1] = int(np.argmin(best))
    for i in range(n - 1, 0, -1):
        j[i - 1] = back[i, j[i]]
    return j, float(np.median(D[np.arange(n), j]))


# --------------------------------------------------------------------------- per feed


def _read_shapes(zip_path: Path, wanted: set[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    from .gtfs import read_table

    with zipfile.ZipFile(zip_path) as zf:
        sh = read_table(zf, "shapes.txt", ["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"], required=False)
    if sh is None or sh.empty:
        return {}
    sh = sh[sh["shape_id"].str.strip().isin(wanted)]
    sh = sh.assign(shape_id=sh["shape_id"].str.strip(), lat=pd.to_numeric(sh["shape_pt_lat"], errors="coerce"),
                   lon=pd.to_numeric(sh["shape_pt_lon"], errors="coerce"),
                   seq=pd.to_numeric(sh["shape_pt_sequence"], errors="coerce")).dropna(subset=["lat", "lon", "seq"])
    sh = sh.sort_values(["shape_id", "seq"], kind="stable")
    out = {}
    for sid, grp in sh.groupby("shape_id", sort=False):
        out[sid] = (grp["lat"].to_numpy(), grp["lon"].to_numpy())
    return out


@dataclass
class FeedRoads:
    """Cumulative metres per road class at each stop position, per pattern."""
    cum: dict[int, np.ndarray]  # pattern index -> (L, 4) metres
    fallback_patterns: int = 0  # no usable shape: stop density is used instead
    # Shape metres by where the lane count came from: tagged, same street,
    # highway-type median, not matched to any road (class copied from neighbours).
    by_source: np.ndarray = field(default_factory=lambda: np.zeros(4))


def _prepare_job(args):
    feed_id, zip_path, wanted = args
    out = {}
    try:
        for sid, (lat, lon) in _read_shapes(Path(zip_path), wanted).items():
            r = resample(lat, lon)
            if r is not None:
                out[sid] = r
    except Exception as e:  # noqa: BLE001 - that feed falls back to stop density
        log.warning("reading shapes of %s failed: %s", feed_id, e)
    return feed_id, out


def classify_feed(feed, prepared: dict, index: RoadIndex) -> FeedRoads:
    out = FeedRoads({})
    matches: dict[str, ShapeMatch] = {}
    sids = list(prepared)
    if sids:
        mids = np.vstack([prepared[s][2] for s in sids])
        hds = np.vstack([prepared[s][3] for s in sids])
        cls, src = index.match(mids, hds)
        bounds = np.r_[0, np.cumsum([len(prepared[s][2]) for s in sids])]
        for sid, a, b in zip(sids, bounds[:-1], bounds[1:]):
            s, pts, _, _, length = prepared[sid]
            m = ShapeMatch(s, pts, _smooth(cls[a:b], length), src[a:b], length)
            matches[sid] = m
            for k in range(3):
                out.by_source[k] += float(length[m.source == k].sum())
            out.by_source[3] += float(length[m.source < 0].sum())
    sxy = project(feed.stops["lat"].to_numpy(), feed.stops["lon"].to_numpy())
    for k, pat in enumerate(feed.patterns):
        mt = matches.get(pat.shape_id)
        if mt is None or (mt.cls == UNMATCHED).all():
            out.fallback_patterns += 1
            continue
        j, off = align_stops(sxy[pat.stops], mt.xy)
        if off > MAX_STOP_OFFSET_M:
            out.fallback_patterns += 1
            continue
        per = np.zeros((len(mt.cls), 4))
        per[np.arange(len(mt.cls)), mt.cls] = mt.length
        cum = np.vstack([np.zeros(4), np.cumsum(per, axis=0)])
        out.cum[k] = cum[j]
    return out


def classify_feeds(feeds, paths: dict[str, str], index_path: Path, cache_dir: Path, workers: int = 3) -> dict[str, FeedRoads]:
    """Road classes for every pattern of every feed (cached per feed)."""
    from concurrent.futures import ProcessPoolExecutor

    from .gtfs import COMPILE_VERSION

    digest = RoadIndex.digest(index_path)
    out: dict[str, FeedRoads] = {}
    todo = []
    for f in feeds:
        zp = Path(paths[f.feed_id])
        cache = cache_dir / f"{f.feed_id}__{zp.stem}__{f.service_date.isoformat()}__c{COMPILE_VERSION}__{digest}.pkl"
        if cache.exists():
            with open(cache, "rb") as fh:
                out[f.feed_id] = pickle.load(fh)
        else:
            todo.append((f, zp, cache))
    if not todo:
        return out
    jobs = [(f.feed_id, str(zp), {p.shape_id for p in f.patterns if p.shape_id}) for f, zp, _ in todo]
    jobs.sort(key=lambda j: -Path(j[1]).stat().st_size)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        prepared = dict(pool.map(_prepare_job, jobs))
    mids = [r[2] for d in prepared.values() for r in d.values()]
    near = cell_keys(np.vstack(mids)) if mids else np.zeros(0, np.int64)
    del mids
    index = RoadIndex.load(index_path, near)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for f, _, cache in todo:
        res = classify_feed(f, prepared.pop(f.feed_id, {}), index)
        with open(cache, "wb") as fh:
            pickle.dump(res, fh, protocol=pickle.HIGHEST_PROTOCOL)
        out[f.feed_id] = res
        log.info("roads %s: %d patterns matched, %d fall back to stop density", f.feed_id, len(res.cum), res.fallback_patterns)
    return out
