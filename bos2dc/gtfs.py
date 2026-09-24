"""GTFS parsing and compilation into per-pattern timetables.

A *pattern* is a unique ordered stop sequence (with pickup/drop-off
permissions) served by one route. Compiling a feed for a service date yields,
for each pattern, the matrix of departure/arrival times of every trip that
runs that day. Everything downstream (frequency graph and itinerary
simulation) works off these matrices.
"""

from __future__ import annotations

import datetime as dt
import io
import logging
import pickle
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .flex import zones_from_gtfs_flex

log = logging.getLogger(__name__)

COMPILE_VERSION = 4
WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


# --------------------------------------------------------------------------- reading


def _open_member(zf: zipfile.ZipFile, name: str):
    # Some feeds nest files in a folder inside the zip.
    for member in zf.namelist():
        if member == name or member.endswith("/" + name):
            return zf.open(member)
    return None


def read_table(zf: zipfile.ZipFile, name: str, usecols=None, required=True) -> pd.DataFrame | None:
    fh = _open_member(zf, name)
    if fh is None:
        if required:
            raise FileNotFoundError(name)
        return None
    raw = fh.read()
    if not raw.strip():
        return None if not required else pd.DataFrame(columns=usecols or [])
    wanted = None if usecols is None else {c for c in usecols}
    df = pd.read_csv(
        io.BytesIO(raw),
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
        skipinitialspace=True,
        on_bad_lines="skip",
        usecols=None if wanted is None else (lambda c: c.strip().lstrip("\ufeff") in wanted),
    )
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    if usecols is not None:
        for c in usecols:
            if c not in df.columns:
                df[c] = ""
        df = df[list(usecols)]
    return df


def parse_times(s: pd.Series) -> np.ndarray:
    """GTFS HH:MM:SS (hours may exceed 24) -> seconds; blanks -> NaN."""
    s = s.astype(str).str.strip()
    parts = s.str.split(":", n=2, expand=True)
    if parts.shape[1] < 3:
        return np.full(len(s), np.nan)
    h = pd.to_numeric(parts[0], errors="coerce")
    m = pd.to_numeric(parts[1], errors="coerce")
    sec = pd.to_numeric(parts[2], errors="coerce")
    return (h * 3600 + m * 60 + sec).to_numpy(dtype=float)


# --------------------------------------------------------------------------- calendars


@dataclass
class Calendar:
    weekly: pd.DataFrame  # service_id, start, end, mon..sun (bool)
    exceptions: pd.DataFrame  # service_id, date, type

    @classmethod
    def load(cls, zf: zipfile.ZipFile) -> "Calendar":
        cal = read_table(zf, "calendar.txt", required=False)
        if cal is None or cal.empty:
            cal = pd.DataFrame(columns=["service_id", "start_date", "end_date", *WEEKDAYS])
        weekly = pd.DataFrame({
            "service_id": cal["service_id"].str.strip(),
            "start": pd.to_datetime(cal["start_date"].str.strip(), format="%Y%m%d", errors="coerce").dt.date,
            "end": pd.to_datetime(cal["end_date"].str.strip(), format="%Y%m%d", errors="coerce").dt.date,
        })
        for d in WEEKDAYS:
            weekly[d] = cal[d].str.strip() == "1" if d in cal else False
        weekly = weekly.dropna(subset=["start", "end"])
        cd = read_table(zf, "calendar_dates.txt", required=False)
        if cd is None or cd.empty:
            cd = pd.DataFrame(columns=["service_id", "date", "exception_type"])
        exceptions = pd.DataFrame({
            "service_id": cd["service_id"].str.strip(),
            "date": pd.to_datetime(cd["date"].str.strip(), format="%Y%m%d", errors="coerce").dt.date,
            "type": pd.to_numeric(cd["exception_type"], errors="coerce"),
        }).dropna(subset=["date"])
        return cls(weekly, exceptions)

    def date_range(self) -> tuple[dt.date, dt.date] | None:
        starts = list(self.weekly["start"]) + list(self.exceptions.loc[self.exceptions["type"] == 1, "date"])
        ends = list(self.weekly["end"]) + list(self.exceptions.loc[self.exceptions["type"] == 1, "date"])
        if not starts:
            return None
        return min(starts), max(ends)

    def services_on(self, day: dt.date) -> set[str]:
        w = self.weekly
        on = set(w.loc[(w["start"] <= day) & (w["end"] >= day) & w[WEEKDAYS[day.weekday()]], "service_id"])
        ex = self.exceptions[self.exceptions["date"] == day]
        on |= set(ex.loc[ex["type"] == 1, "service_id"])
        on -= set(ex.loc[ex["type"] == 2, "service_id"])
        return on


def service_date_range(zip_path) -> tuple[dt.date, dt.date] | None:
    with zipfile.ZipFile(zip_path) as zf:
        return Calendar.load(zf).date_range()


def choose_service_date(cal: Calendar, trips_per_service: pd.Series, target: dt.date) -> tuple[dt.date | None, str]:
    """Pick the date whose timetable stands in for `target`.

    Uses `target` itself when the feed covers it; otherwise (stale or
    not-yet-valid feeds) the nearest date with the same weekday. Candidates
    that look like holidays (under 80% of the busiest same-weekday date's
    trips) are skipped.
    """
    rng = cal.date_range()
    if rng is None:
        return None, "no calendar"
    lo, hi = rng
    k_min = -((target - lo).days // 7)  # ceil((lo - target) / 7 days)
    k_max = (hi - target).days // 7
    if k_min > k_max:
        return None, "no date with matching weekday"
    k0 = min(max(0, k_min), k_max)
    ks = sorted(range(max(k_min, k0 - 8), min(k_max, k0 + 8) + 1), key=lambda k: abs(k))[:8]
    candidates = [target + dt.timedelta(weeks=k) for k in ks]
    counts = [int(trips_per_service.reindex(list(cal.services_on(d))).fillna(0).sum()) for d in candidates]
    best = max(counts)
    if best == 0:
        return None, f"no service on any {WEEKDAYS[target.weekday()]}"
    for d, n in zip(candidates, counts):
        if n >= 0.8 * best:
            note = "" if d == target else f"timetable of {d.isoformat()} used for {target.isoformat()}"
            return d, note
    raise AssertionError("unreachable")


# --------------------------------------------------------------------------- compiled model


@dataclass
class Pattern:
    route_id: str
    route_name: str
    agency: str
    headsign: str
    stops: np.ndarray  # int32 stop index into CompiledFeed.stops, length L
    board: np.ndarray  # bool, pickup allowed
    alight: np.ndarray  # bool, drop-off allowed
    dep: np.ndarray  # int32 seconds after midnight of the service day, (n_trips, L)
    arr: np.ndarray  # int32, (n_trips, L)


@dataclass
class CompiledFeed:
    feed_id: str
    provider: str
    service_date: dt.date
    stops: pd.DataFrame  # stop_id, name, lat, lon (row position = stop index)
    patterns: list[Pattern] = field(default_factory=list)
    flex_zones: list = field(default_factory=list)  # flex.FlexZone from GTFS-Flex data
    notes: list[str] = field(default_factory=list)


def _interpolate_times(times: np.ndarray) -> np.ndarray:
    """Fill non-timepoint stops linearly. Rows are grouped by trip and each
    trip's first and last stop carry times (GTFS requirement), so a global
    interpolation over row position never mixes trips."""
    known = ~np.isnan(times)
    if known.all():
        return times
    idx = np.arange(len(times))
    out = times.copy()
    out[~known] = np.interp(idx[~known], idx[known], times[known])
    return out


def compile_feed(zip_path, feed_id: str, provider: str, target: dt.date) -> CompiledFeed | None:
    agency_re = config.compile_patterns(config.EXCLUDED_AGENCY_PATTERNS)
    with zipfile.ZipFile(zip_path) as zf:
        agency = read_table(zf, "agency.txt", ["agency_id", "agency_name"], required=False)
        routes = read_table(zf, "routes.txt", ["route_id", "agency_id", "route_short_name", "route_long_name", "route_type"])
        trips = read_table(zf, "trips.txt", ["route_id", "service_id", "trip_id", "trip_headsign"])
        cal = Calendar.load(zf)

        routes["route_type"] = pd.to_numeric(routes["route_type"], errors="coerce")
        routes = routes[routes["route_type"].isin(config.BUS_ROUTE_TYPES)].copy()
        agency_names = {}
        if agency is not None and not agency.empty:
            agency_names = dict(zip(agency["agency_id"], agency["agency_name"]))
            default_agency = agency["agency_name"].iloc[0]
        else:
            default_agency = provider
        routes["agency"] = routes["agency_id"].map(agency_names).fillna(default_agency)
        routes.loc[routes["agency"] == "", "agency"] = default_agency
        routes = routes[~routes["agency"].str.contains(agency_re)]
        if routes.empty:
            return None
        trips = trips[trips["route_id"].isin(routes["route_id"])].drop_duplicates("trip_id")
        routes = routes.drop_duplicates("route_id")
        trips_per_service = trips.groupby("service_id").size()
        day, note = choose_service_date(cal, trips_per_service, target)
        if day is None:
            log.info("%s: %s", feed_id, note)
            return None
        trips = trips[trips["service_id"].isin(cal.services_on(day))]
        if trips.empty:
            return None

        freq = read_table(zf, "frequencies.txt", ["trip_id", "start_time", "end_time", "headway_secs"], required=False)
        st = read_table(zf, "stop_times.txt", ["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence", "pickup_type", "drop_off_type",
                                                "location_id", "start_pickup_drop_off_window", "end_pickup_drop_off_window"])
        stops = read_table(zf, "stops.txt", ["stop_id", "stop_name", "stop_lat", "stop_lon"])
        st = st[st["trip_id"].isin(trips["trip_id"])]
        flex_zones = zones_from_gtfs_flex(zf, st, trips, routes, day, feed_id)

    st = st[st["stop_id"] != ""]
    st = st.assign(seq=pd.to_numeric(st["stop_sequence"], errors="coerce")).dropna(subset=["seq"])
    st = st.sort_values(["trip_id", "seq"], kind="stable")

    stops = stops.assign(
        lat=pd.to_numeric(stops["stop_lat"], errors="coerce"),
        lon=pd.to_numeric(stops["stop_lon"], errors="coerce"),
    ).dropna(subset=["lat", "lon"])
    stops = stops[(stops["lat"].abs() > 1) & (stops["lon"].abs() > 1)]
    stops = stops.drop_duplicates("stop_id")
    st = st[st["stop_id"].isin(stops["stop_id"])]
    if st.empty:
        return CompiledFeed(feed_id, provider, day, stops.iloc[:0][["stop_id", "stop_name", "lat", "lon"]].rename(columns={"stop_name": "name"}),
                            flex_zones=flex_zones) if flex_zones else None

    stops = stops[stops["stop_id"].isin(st["stop_id"].unique())].reset_index(drop=True)
    stop_code = pd.Series(np.arange(len(stops), dtype=np.int64), index=stops["stop_id"])

    arr = parse_times(st["arrival_time"])
    dep = parse_times(st["departure_time"])
    arr = np.where(np.isnan(arr), dep, arr)
    dep = np.where(np.isnan(dep), arr, dep)
    # Trips missing first/last times would poison interpolation; drop them.
    trip_ids = st["trip_id"].to_numpy()
    first = np.r_[True, trip_ids[1:] != trip_ids[:-1]]
    last = np.r_[trip_ids[1:] != trip_ids[:-1], True]
    bad_trips = set(trip_ids[(first | last) & np.isnan(dep)])
    if bad_trips:
        keep = ~np.isin(trip_ids, list(bad_trips))
        st, arr, dep = st[keep], arr[keep], dep[keep]
        trip_ids = st["trip_id"].to_numpy()
        first = np.r_[True, trip_ids[1:] != trip_ids[:-1]] if len(trip_ids) else first[:0]
    if not len(st):
        return None
    arr = _interpolate_times(arr)
    dep = _interpolate_times(dep)

    code = stop_code.loc[st["stop_id"]].to_numpy() * 4
    code += np.where(st["pickup_type"].str.strip() == "1", 0, 2)
    code += np.where(st["drop_off_type"].str.strip() == "1", 0, 1)

    starts = np.flatnonzero(first)
    lengths = np.diff(np.r_[starts, len(st)])
    trip_order = trip_ids[starts]
    route_of_trip = trips.set_index("trip_id")["route_id"].loc[trip_order].to_numpy()
    # Pattern key: route plus the trip's full (stop, pickup, drop-off) sequence.
    keys = [f"{r}\x00".encode() + code[s:s + n].tobytes() for r, s, n in zip(route_of_trip, starts, lengths)]
    pat_ids, uniq = pd.factorize(pd.Series(keys, dtype=object))

    tinfo = trips.set_index("trip_id")
    rinfo = routes.set_index("route_id")
    freq_by_trip: dict[str, list[tuple[float, float, float]]] = {}
    if freq is not None and not freq.empty:
        for r in freq.itertuples(index=False):
            s0, s1 = parse_times(pd.Series([r.start_time, r.end_time]))
            h = pd.to_numeric(r.headway_secs, errors="coerce")
            if np.isfinite(s0) and np.isfinite(s1) and h and h > 0:
                freq_by_trip.setdefault(r.trip_id, []).append((s0, s1, float(h)))

    out = CompiledFeed(feed_id=feed_id, provider=provider, service_date=day, stops=stops[["stop_id", "stop_name", "lat", "lon"]].rename(columns={"stop_name": "name"}),
                       flex_zones=flex_zones)
    if note:
        out.notes.append(note)
    order = np.argsort(pat_ids, kind="stable")
    bounds = np.flatnonzero(np.r_[True, pat_ids[order][1:] != pat_ids[order][:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        members = order[a:b]
        L = lengths[members[0]]
        if L < 2:
            continue
        idx = starts[members][:, None] + np.arange(L)
        d_mat, a_mat = dep[idx], arr[idx]
        tids = trip_order[members]
        if freq_by_trip:
            rows_d, rows_a = [], []
            for i, t in enumerate(tids):
                if t in freq_by_trip:
                    rel_d, rel_a = d_mat[i] - d_mat[i, 0], a_mat[i] - d_mat[i, 0]
                    for s0, s1, h in freq_by_trip[t]:
                        for t0 in np.arange(s0, s1, h):
                            rows_d.append(rel_d + t0)
                            rows_a.append(rel_a + t0)
                else:
                    rows_d.append(d_mat[i])
                    rows_a.append(a_mat[i])
            d_mat, a_mat = np.array(rows_d), np.array(rows_a)
        # Duplicate trips (overlapping service_ids in some feeds) would
        # inflate frequency: keep one trip per first departure time.
        _, keep = np.unique(d_mat[:, 0], return_index=True)
        keep.sort()
        d_mat, a_mat = d_mat[keep], a_mat[keep]
        s0, n0 = starts[members[0]], lengths[members[0]]
        c = code[s0:s0 + n0]
        t0 = tinfo.loc[tids[0]]
        r = rinfo.loc[t0["route_id"]]
        name = (r["route_short_name"] or "").strip() or (r["route_long_name"] or "").strip() or t0["route_id"]
        if r["route_short_name"] and r["route_long_name"]:
            name = f"{r['route_short_name'].strip()} {r['route_long_name'].strip()}"
        out.patterns.append(Pattern(
            route_id=t0["route_id"],
            route_name=name,
            agency=r["agency"],
            headsign=(t0["trip_headsign"] or "").strip(),
            stops=(c // 4).astype(np.int32),
            board=(c & 2).astype(bool),
            alight=(c & 1).astype(bool),
            dep=np.round(d_mat).astype(np.int32),
            arr=np.round(a_mat).astype(np.int32),
        ))
    return out if out.patterns or out.flex_zones else None


def compile_cached(zip_path, feed_id: str, provider: str, target: dt.date, cache_dir: Path) -> CompiledFeed | None:
    zp = Path(zip_path)
    key = f"{feed_id}__{zp.stem}__{target.isoformat()}__v{COMPILE_VERSION}.pkl"
    path = cache_dir / key
    if path.exists():
        with open(path, "rb") as fh:
            return pickle.load(fh)
    feed = compile_feed(zp, feed_id, provider, target)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(feed, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return feed
