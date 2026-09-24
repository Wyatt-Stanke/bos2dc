"""Mobility Database catalog access: feed discovery, selection and download."""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import requests

from . import atlas, config
from .flex import fetch_zones
from .geo import bbox_near_polyline

log = logging.getLogger(__name__)


class MobilityDatabase:
    """Minimal client for the Mobility Database Catalog API (OAuth2 refresh flow)."""

    def __init__(self, refresh_token: str, base: str = config.API_BASE):
        if not refresh_token:
            raise ValueError("A Mobility Database refresh token is required (set $REFRESH_TOKEN)")
        self.base = base
        self.refresh_token = refresh_token
        self.session = requests.Session()
        self._access_token: str | None = None
        self._expires_at = 0.0

    def _token(self) -> str:
        # Access tokens last one hour; renew with a minute of margin.
        if self._access_token is None or time.time() > self._expires_at - 60:
            r = self.session.post(f"{self.base}/tokens", json={"refresh_token": self.refresh_token}, timeout=30)
            r.raise_for_status()
            self._access_token = r.json()["access_token"]
            self._expires_at = time.time() + 3600
        return self._access_token

    def get(self, path: str, **params):
        for attempt in range(4):
            r = self.session.get(
                f"{self.base}{path}",
                params=params,
                headers={"Authorization": f"Bearer {self._token()}"},
                timeout=60,
            )
            if r.status_code == 401:
                self._access_token = None
            elif r.status_code < 500:
                r.raise_for_status()
                return r.json()
            time.sleep(2 ** attempt)
        r.raise_for_status()
        return r.json()

    def gtfs_feeds(self, **params) -> list[dict]:
        out, offset, limit = [], 0, 2500
        while True:
            page = self.get("/gtfs_feeds", limit=limit, offset=offset, **params)
            out.extend(page)
            if len(page) < limit:
                return out
            offset += limit


@dataclass
class FeedChoice:
    feed_id: str
    provider: str
    feed_name: str
    status: str
    dataset_id: str | None
    url: str
    source: str  # "catalog" (Mobility Database hosted copy) or "producer"
    service_start: str | None
    service_end: str | None
    stale: bool
    bbox: dict | None = None
    producer_url: str | None = None
    producer_auth: int = 0
    notes: list[str] = field(default_factory=list)
    path: str | None = None  # local zip path once downloaded


def _norm_name(provider: str, feed_name: str) -> str:
    s = f"{provider} {feed_name}".lower()
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def discover_feeds(api: MobilityDatabase) -> list[dict]:
    """All GTFS feeds located in a corridor state or overlapping the corridor bbox."""
    feeds: dict[str, dict] = {}
    for state in config.CORRIDOR_STATES:
        for f in api.gtfs_feeds(country_code="US", subdivision_name=state):
            feeds[f["id"]] = f
    lats = [p[0] for p in config.CORRIDOR_SPINE]
    lons = [p[1] for p in config.CORRIDOR_SPINE]
    box = dict(
        dataset_latitudes=f"{min(lats) - 0.5},{max(lats) + 0.5}",
        dataset_longitudes=f"{min(lons) - 0.6},{max(lons) + 0.6}",
    )
    for method in ("completely_enclosed", "partially_enclosed"):
        for f in api.gtfs_feeds(bounding_filter_method=method, **box):
            feeds[f["id"]] = f
    log.info("catalog: %d candidate feeds", len(feeds))
    return list(feeds.values())


def select_feeds(feeds: list[dict], today: dt.date, buffer_km: float = config.CORRIDOR_BUFFER_KM) -> tuple[list[FeedChoice], list[tuple[str, str]]]:
    """Filter the catalog down to local bus feeds near the corridor.

    Returns (choices, rejected) where rejected is a list of (feed label, reason).
    """
    prov_re = config.compile_patterns(config.EXCLUDED_PROVIDER_PATTERNS)
    name_re = config.compile_patterns(config.EXCLUDED_FEED_NAME_PATTERNS)
    rejected: list[tuple[str, str]] = []
    kept: list[FeedChoice] = []
    for f in feeds:
        label = f"{f['id']} {f.get('provider', '')} {f.get('feed_name') or ''}".strip()
        ld = f.get("latest_dataset") or {}
        if f.get("status") not in config.USABLE_STATUSES:
            rejected.append((label, f"status={f.get('status')}"))
            continue
        if prov_re.search(f.get("provider") or ""):
            rejected.append((label, "provider excluded (not fixed-fare local bus)"))
            continue
        if name_re.search(f.get("feed_name") or ""):
            rejected.append((label, "feed name excluded (rail/ferry/flex)"))
            continue
        if not ld.get("hosted_url"):
            rejected.append((label, "no hosted dataset"))
            continue
        bb = ld.get("bounding_box")
        if bb and None not in bb.values() and not bbox_near_polyline(
            bb["minimum_latitude"], bb["maximum_latitude"], bb["minimum_longitude"], bb["maximum_longitude"],
            config.CORRIDOR_SPINE, buffer_km * 1000,
        ):
            rejected.append((label, "outside corridor"))
            continue
        end = ld.get("service_date_range_end")
        start = ld.get("service_date_range_start")
        stale = bool(end) and dt.date.fromisoformat(end[:10]) < today
        kept.append(FeedChoice(
            feed_id=f["id"],
            provider=f.get("provider") or "",
            feed_name=f.get("feed_name") or "",
            status=f["status"],
            dataset_id=ld.get("id"),
            url=ld["hosted_url"],
            source="catalog",
            service_start=start[:10] if start else None,
            service_end=end[:10] if end else None,
            stale=stale,
            bbox=bb,
            producer_url=(f.get("source_info") or {}).get("producer_url"),
            producer_auth=(f.get("source_info") or {}).get("authentication_type") or 0,
        ))

    # Same agency listed more than once (mdb/ntd/tld mirrors): drop inactive
    # copies when a maintained one exists, and exact-duplicate datasets.
    groups: dict[str, list[FeedChoice]] = {}
    for c in kept:
        groups.setdefault(_norm_name(c.provider, c.feed_name), []).append(c)
    out: list[FeedChoice] = []
    for members in groups.values():
        live = [c for c in members if c.status != "inactive"]
        for c in members:
            if c.status == "inactive" and live:
                rejected.append((f"{c.feed_id} {c.provider}", f"inactive duplicate of {live[0].feed_id}"))
            else:
                out.append(c)
    return sorted(out, key=lambda c: c.feed_id), rejected


def _download(session: requests.Session, url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    for attempt in range(4):
        try:
            with session.get(url, stream=True, timeout=(30, 300)) as r:
                r.raise_for_status()
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
            tmp.rename(dest)
            return
        except requests.RequestException as e:
            client_error = isinstance(e, requests.HTTPError) and e.response is not None and e.response.status_code < 500
            if attempt == 3 or client_error:
                raise
            time.sleep(2 ** (attempt + 1))


def _stops_bbox(zip_path) -> tuple[float, float, float, float] | None:
    import zipfile

    import numpy as np
    import pandas as pd

    from .gtfs import read_table

    try:
        with zipfile.ZipFile(zip_path) as zf:
            s = read_table(zf, "stops.txt", ["stop_lat", "stop_lon"])
    except Exception:  # noqa: BLE001
        return None
    lat = pd.to_numeric(s["stop_lat"], errors="coerce").to_numpy(dtype=float)
    lon = pd.to_numeric(s["stop_lon"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) > 1)
    if not ok.any():
        return None
    return lat[ok].min(), lat[ok].max(), lon[ok].min(), lon[ok].max()


def _overlaps(a, b, pad: float = 0.3) -> bool:
    return not (a[1] + pad < b[0] or b[1] + pad < a[0] or a[3] + pad < b[2] or b[3] + pad < a[2])


def _try_fresh(session, c: FeedChoice, url: str, dest: Path, today: dt.date, label: str, trusted: bool = True) -> bool:
    """Download `url`; adopt it for `c` if its calendar reaches `today` and it
    covers the same area as the stale copy (guards against name matches that
    are a different agency with the same name)."""
    from .gtfs import service_date_range

    try:
        if not dest.exists():
            log.info("stale %s; trying %s %s", c.feed_id, label, url)
            _download(session, url, dest)
        rng = service_date_range(dest)
    except Exception as e:  # noqa: BLE001 - any failure just means "keep the catalog copy"
        log.info("%s for %s unusable: %s", label, c.feed_id, e)
        return False
    if not rng or rng[1] < today:
        return False
    old_box, new_box = _stops_bbox(c.path), _stops_bbox(dest)
    if old_box is None and c.bbox and None not in c.bbox.values():
        old_box = (c.bbox["minimum_latitude"], c.bbox["maximum_latitude"], c.bbox["minimum_longitude"], c.bbox["maximum_longitude"])
    if (old_box is None and not trusted) or (old_box and new_box and not _overlaps(old_box, new_box)):
        log.info("%s for %s: cannot confirm it covers the same area; ignored", label, c.feed_id)
        return False
    c.path, c.url, c.source = str(dest), url, label
    c.service_start, c.service_end = rng[0].isoformat(), rng[1].isoformat()
    c.stale = False
    c.notes.append(f"catalog copy stale; using {label} URL")
    return True


def download_feeds(choices: list[FeedChoice], data_dir: Path, today: dt.date, try_producer: bool = True,
                   atlas_feeds: list | None = None) -> list[FeedChoice]:
    """Download each chosen dataset (cached by dataset id).

    For stale catalog datasets, the agency's own producer URL and then any
    matching Transitland Atlas URLs are tried; the first whose calendar
    reaches `today` replaces the catalog copy.
    """
    from . import atlas
    from .gtfs import service_date_range  # local import: gtfs pulls in pandas

    session = requests.Session()
    feed_root = data_dir / "feeds"
    ready: list[FeedChoice] = []
    for c in choices:
        dest = feed_root / c.feed_id / f"{c.dataset_id}.zip"
        try:
            if not dest.exists():
                log.info("download %s %s", c.feed_id, c.provider)
                _download(session, c.url, dest)
        except requests.RequestException as e:
            log.warning("skip %s: %s", c.feed_id, e)
            continue
        c.path = str(dest)
        if c.service_end is None or c.service_start is None:
            rng = service_date_range(dest)
            if rng:
                c.service_start, c.service_end = (d.isoformat() for d in rng)
                c.stale = rng[1] < today
        if c.stale and try_producer and c.producer_url and not c.producer_auth:
            _try_fresh(session, c, c.producer_url, feed_root / c.feed_id / f"producer-{today.isoformat()}.zip", today, "producer")
        if c.stale and atlas_feeds:
            for k, (af, by_url) in enumerate(atlas.match(atlas_feeds, [c.producer_url, c.url], c.provider)):
                dest = feed_root / c.feed_id / f"atlas-{k}-{today.isoformat()}.zip"
                if _try_fresh(session, c, af.url, dest, today, f"transitland-atlas {af.onestop_id}", trusted=by_url):
                    break
        if c.stale:
            c.notes.append(f"stale: service ends {c.service_end}; schedules mapped onto same weekday")
        ready.append(c)
    return ready


def atlas_extras(atlas_feeds: list, ids: list[str], data_dir: Path, today: dt.date) -> list[FeedChoice]:
    """Feeds taken from the Transitland Atlas because the catalog lacks them."""
    from .gtfs import service_date_range

    session = requests.Session()
    out = []
    by_id = {f.onestop_id: f for f in atlas_feeds}
    for oid in ids:
        af = by_id.get(oid)
        if af is None or not af.url:
            log.warning("atlas feed %s not found", oid)
            continue
        dest = data_dir / "feeds" / oid / f"atlas-{today.isoformat()}.zip"
        try:
            if not dest.exists():
                log.info("download atlas %s", oid)
                _download(session, af.url, dest)
            rng = service_date_range(dest)
        except Exception as e:  # noqa: BLE001
            log.warning("skip atlas feed %s: %s", oid, e)
            continue
        out.append(FeedChoice(
            feed_id=oid, provider=" / ".join(af.operators) or oid, feed_name="", status="atlas",
            dataset_id=dest.stem, url=af.url, source="atlas-extra",
            service_start=rng[0].isoformat() if rng else None, service_end=rng[1].isoformat() if rng else None,
            stale=bool(rng) and rng[1] < today, path=str(dest),
        ))
    return out


def write_manifest(choices: list[FeedChoice], rejected, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "feeds": [asdict(c) for c in choices],
        "rejected": [{"feed": a, "reason": b} for a, b in rejected],
    }
    path.write_text(json.dumps(payload, indent=1))


def read_manifest(path: Path) -> list[FeedChoice]:
    data = json.loads(path.read_text())
    return [FeedChoice(**f) for f in data["feeds"]]


def fetch(data_dir: Path, today: dt.date, refresh_token: str | None = None, buffer_km: float = config.CORRIDOR_BUFFER_KM,
          use_atlas: bool = True) -> list[FeedChoice]:
    api = MobilityDatabase(refresh_token or os.environ.get("REFRESH_TOKEN", ""))
    feeds = discover_feeds(api)
    (data_dir / "catalog.json").parent.mkdir(parents=True, exist_ok=True)
    (data_dir / "catalog.json").write_text(json.dumps(feeds))
    choices, rejected = select_feeds(feeds, today, buffer_km)
    log.info("selected %d feeds (%d rejected)", len(choices), len(rejected))
    atlas_feeds = atlas.load(data_dir, config.ATLAS_FALLBACK_FILES) if use_atlas else []
    ready = download_feeds(choices, data_dir, today, atlas_feeds=atlas_feeds)
    if use_atlas:
        have = {c.feed_id for c in ready}
        ready += [c for c in atlas_extras(atlas_feeds, config.ATLAS_EXTRA_FEEDS, data_dir, today) if c.feed_id not in have]
    write_manifest(ready, rejected, data_dir / "manifest.json")
    zones = fetch_zones(config.FLEX_ARCGIS_SOURCES, data_dir / "flex_zones.geojson")
    log.info("flex zones: %d", len(zones))
    return ready
