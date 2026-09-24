"""Demand-response ("Flex") zones.

A Flex zone has no timetable: during its service hours a rider books a trip
between any two points inside the polygon, at the agency's regular fare. For
routing it is modelled as

* an *access point* at every fixed-route stop inside the polygon, plus a
  virtual *zone-edge* point on the boundary nearest to each stop just
  outside it (a Flex drop-off at the town line followed by a walk);
* a Flex edge between every pair of access points, with in-vehicle time from
  straight-line distance and an assumed road speed;
* an equivalent frequency of `service span / flex_headway`: the zone is
  always available, but each boarding costs a response wait comparable to a
  bus with that headway.

Zones come from two places: GTFS-Flex v2 data inside catalog feeds
(locations.geojson + stop_times pickup/drop-off windows) and agencies that
publish their zones only as an ArcGIS feature layer (RIPTA).
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .geo import project

log = logging.getLogger(__name__)

WEEKDAY_KEYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


@dataclass
class FlexZone:
    zone_id: str
    name: str
    agency: str
    # Polygon rings as (n, 2) arrays of lon, lat. Membership uses the
    # even-odd rule over all rings, which handles holes and multipolygons.
    rings: list[np.ndarray]
    # weekday (0 = Monday) -> list of (start_s, end_s, headway_s or None)
    service: dict[int, list[tuple[int, int, int | None]]]
    source: str = ""
    info_url: str = ""
    notes: list[str] = field(default_factory=list)

    def windows(self, day: dt.date) -> list[tuple[int, int, int | None]]:
        return self.service.get(day.weekday(), [])

    def bbox(self) -> tuple[float, float, float, float]:
        pts = np.vstack(self.rings)
        return pts[:, 1].min(), pts[:, 1].max(), pts[:, 0].min(), pts[:, 0].max()

    def to_feature(self) -> dict:
        return {
            "type": "Feature",
            "geometry": {"type": "MultiPolygon", "coordinates": [[r.tolist()] for r in self.rings]},
            "properties": {
                "zone_id": self.zone_id, "name": self.name, "agency": self.agency,
                "service": {WEEKDAY_KEYS[k]: [list(w) for w in v] for k, v in self.service.items()},
                "source": self.source, "info_url": self.info_url, "notes": self.notes,
            },
        }

    @classmethod
    def from_feature(cls, f: dict) -> "FlexZone":
        p = f["properties"]
        return cls(
            zone_id=p["zone_id"], name=p["name"], agency=p.get("agency", ""),
            rings=_rings(f["geometry"]),
            service={WEEKDAY_KEYS.index(k): [tuple(w) for w in v] for k, v in p.get("service", {}).items()},
            source=p.get("source", ""), info_url=p.get("info_url", ""), notes=list(p.get("notes", [])),
        )


def _rings(geom: dict) -> list[np.ndarray]:
    if geom["type"] == "Polygon":
        return [np.asarray(r, dtype=float)[:, :2] for r in geom["coordinates"]]
    if geom["type"] == "MultiPolygon":
        return [np.asarray(r, dtype=float)[:, :2] for poly in geom["coordinates"] for r in poly]
    raise ValueError(f"unsupported geometry {geom['type']}")


# --------------------------------------------------------------------------- geometry


def contains(zone: FlexZone, lat, lon) -> np.ndarray:
    """Even-odd point-in-polygon test (vectorised over points)."""
    x = np.asarray(lon, dtype=float)
    y = np.asarray(lat, dtype=float)
    inside = np.zeros(len(x), dtype=bool)
    for ring in zone.rings:
        x1, y1 = ring[:-1, 0], ring[:-1, 1]
        x2, y2 = ring[1:, 0], ring[1:, 1]
        for a, b, c, d in zip(x1, y1, x2, y2):
            crosses = (b > y) != (d > y)
            xint = a + (y - b) * (c - a) / np.where(d == b, 1e-12, d - b)
            inside ^= crosses & (x < xint)
    return inside


def nearest_boundary(zone: FlexZone, lat, lon) -> tuple[np.ndarray, np.ndarray]:
    """Distance (m) from each point to the zone boundary and the nearest
    boundary point (projected metres)."""
    pts = project(lat, lon)
    best_d = np.full(len(pts), np.inf)
    best_q = np.zeros_like(pts)
    for ring in zone.rings:
        xy = project(ring[:, 1], ring[:, 0])
        a, ab = xy[:-1], xy[1:] - xy[:-1]
        denom = np.maximum((ab * ab).sum(1), 1e-9)
        # (points x segments) is small: zones have hundreds of vertices and
        # this is called on a pre-filtered set of nearby stops.
        t = np.clip(((pts[:, None, :] - a[None]) * ab[None]).sum(2) / denom[None], 0, 1)
        q = a[None] + t[..., None] * ab[None]
        d = np.linalg.norm(q - pts[:, None, :], axis=2)
        k = d.argmin(axis=1)
        dk = d[np.arange(len(pts)), k]
        better = dk < best_d
        best_d[better] = dk[better]
        best_q[better] = q[np.arange(len(pts)), k][better]
    return best_d, best_q


# --------------------------------------------------------------------------- ArcGIS sources

_HOURS_RE = re.compile(
    r"(\d{1,2})(?::(\d{2}))?\s*([ap])\.?m\.?\s*[-–]\s*(\d{1,2})(?::(\d{2}))?\s*([ap])\.?m\.?(?:[^\d]*every\s*(\d+)\s*min)?",
    re.IGNORECASE,
)


def parse_hours(text: str | None) -> list[tuple[int, int, int | None]]:
    """'6:00am-6:30pm' / '6:45am-5:12pm: every 90 min' / 'NO SERVICE'."""
    out = []
    for m in _HOURS_RE.finditer(text or ""):
        h1, m1, p1, h2, m2, p2, every = m.groups()

        def secs(h, mi, p):
            h = int(h) % 12 + (12 if p.lower() == "p" else 0)
            return h * 3600 + int(mi or 0) * 60

        out.append((secs(h1, m1, p1), secs(h2, m2, p2), int(every) * 60 if every else None))
    return out


def load_arcgis_zones(src: dict, session: requests.Session | None = None) -> list[FlexZone]:
    session = session or requests.Session()
    r = session.get(f"{src['url']}/query", params={"where": "1=1", "outFields": "*", "outSR": 4326, "f": "geojson"}, timeout=60)
    r.raise_for_status()
    zones = []
    for f in r.json().get("features", []):
        p = f["properties"]
        if src.get("type_field") and p.get(src["type_field"]) not in src.get("keep_types", [p.get(src["type_field"])]):
            continue
        service = {}
        for key, fieldname in src["hours_fields"].items():
            windows = parse_hours(p.get(fieldname))
            for wd in {"weekday": range(5), "saturday": [5], "sunday": [6]}[key]:
                if windows:
                    service[wd] = windows
        zones.append(FlexZone(
            zone_id=f"{src['agency_id']}-{p.get(src['id_field'])}",
            name=str(p.get(src["name_field"])),
            agency=src["agency"],
            rings=_rings(f["geometry"]),
            service=service,
            source=src["url"],
            info_url=p.get(src.get("info_field", ""), "") or "",
        ))
    return zones


def fetch_zones(sources: list[dict], dest: Path) -> list[FlexZone]:
    zones: list[FlexZone] = []
    for src in sources:
        try:
            zones += load_arcgis_zones(src)
        except requests.RequestException as e:
            log.warning("flex zones from %s unavailable: %s", src["agency"], e)
    if zones or not dest.exists():
        write_zones(zones, dest)
    return read_zones(dest)


def write_zones(zones: list[FlexZone], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [z.to_feature() for z in zones]}))


def read_zones(path: Path) -> list[FlexZone]:
    if not path.exists():
        return []
    return [FlexZone.from_feature(f) for f in json.loads(path.read_text())["features"]]


# --------------------------------------------------------------------------- GTFS-Flex v2


def _parse_times(values) -> tuple[float, float]:
    out = []
    for v in values:
        parts = str(v).strip().split(":")
        try:
            out.append(int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2]))
        except (ValueError, IndexError):
            out.append(float("nan"))
    return tuple(out)


def zones_from_gtfs_flex(zf: zipfile.ZipFile, st: pd.DataFrame, trips: pd.DataFrame, routes: pd.DataFrame,
                         day: dt.date, feed_id: str) -> list[FlexZone]:
    """Zones served on `day` by GTFS-Flex trips (stop_times rows that
    reference a locations.geojson feature instead of a stop).

    Each zone gets the union of the pickup/drop-off windows of the trips that
    serve it. Trips linking two different zones are approximated as service
    within each zone (inter-zone travel is not modelled).
    """
    member = next((m for m in zf.namelist() if m.endswith("locations.geojson")), None)
    if member is None or "location_id" not in st.columns:
        return []
    locations = {f["id"] if "id" in f else f["properties"].get("id"): f for f in json.loads(zf.read(member))["features"]}
    rows = st[st["location_id"].astype(str).str.len() > 0]
    if rows.empty:
        return []
    rows = rows.merge(trips[["trip_id", "route_id"]], on="trip_id").merge(routes[["route_id", "route_short_name", "route_long_name", "agency"]], on="route_id")
    zones: dict[str, FlexZone] = {}
    wd = day.weekday()
    for r in rows.itertuples(index=False):
        loc = locations.get(r.location_id)
        if loc is None or loc.get("geometry", {}).get("type") not in ("Polygon", "MultiPolygon"):
            continue
        s0, s1 = _parse_times([r.start_pickup_drop_off_window, r.end_pickup_drop_off_window])
        if not (np.isfinite(s0) and np.isfinite(s1)):
            continue
        z = zones.get(r.location_id)
        if z is None:
            name = (r.route_short_name or r.route_long_name or r.location_id).strip()
            z = zones[r.location_id] = FlexZone(f"{feed_id}-{r.location_id}", name, r.agency, _rings(loc["geometry"]), {wd: []}, source=feed_id)
        z.service[wd].append((int(s0), int(s1), None))
    for z in zones.values():
        z.service[wd] = _union(z.service[wd])
    return list(zones.values())


def _union(windows):
    out = []
    for s, e, h in sorted(windows):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e), out[-1][2])
        else:
            out.append((s, e, h))
    return out
