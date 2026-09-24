"""Transitland Atlas (github.com/transitland/transitland-atlas) as a second
feed source.

The Mobility Database lags for several agencies on the corridor (GATRA, WRTA,
Bee-Line and Harford Transit are listed as inactive with expired datasets)
and has no entry at all for some others (Northeastern Connecticut Transit
District, Windham Region Transit District, several New Jersey county
systems). The atlas is a git repository of DMFR records that tracks agencies'
current download URLs. It is used two ways:

* stale catalog feeds are matched to atlas records (by any URL the two share,
  then by operator name) and the atlas' `static_current` URL is tried;
* `config.ATLAS_EXTRA_FEEDS` adds feeds that the catalog does not have.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import requests

log = logging.getLogger(__name__)

REPO = "https://github.com/transitland/transitland-atlas"
RAW = "https://raw.githubusercontent.com/transitland/transitland-atlas/main/feeds/"


@dataclass
class AtlasFeed:
    onestop_id: str
    url: str | None
    historic: list[str] = field(default_factory=list)
    operators: list[str] = field(default_factory=list)

    def all_urls(self) -> set[str]:
        return {norm_url(u) for u in [self.url, *self.historic] if u}


def norm_url(u: str | None) -> str:
    return re.sub(r"^https?://(www\.)?", "", (u or "").strip()).rstrip("/").lower()


def norm_name(s: str) -> str:
    s = re.sub(r"\(.*?\)", " ", s.lower())
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s).split())


def _sync_repo(dest: Path) -> Path | None:
    """Shallow clone (or fast-forward) the atlas; None if git is unavailable."""
    if not shutil.which("git"):
        return None
    try:
        if (dest / ".git").exists():
            subprocess.run(["git", "-C", str(dest), "pull", "--ff-only", "--depth", "1"], check=True, capture_output=True, timeout=600)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "--depth", "1", REPO, str(dest)], check=True, capture_output=True, timeout=900,
                           env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"})
        return dest / "feeds"
    except (subprocess.SubprocessError, OSError) as e:
        log.warning("could not sync transitland-atlas: %s", e)
        return dest / "feeds" if (dest / "feeds").exists() else None


def _parse(records: list[dict]) -> list[AtlasFeed]:
    out = []
    for d in records:
        op_names = {o.get("onestop_id"): o.get("name") for o in d.get("operators", [])}
        feed_ops: dict[str, list[str]] = {}
        for o in d.get("operators", []):
            for af in o.get("associated_feeds", []):
                if af.get("feed_onestop_id"):
                    feed_ops.setdefault(af["feed_onestop_id"], []).append(o.get("name") or "")
        for f in d.get("feeds", []):
            if f.get("spec") != "gtfs":
                continue
            urls = f.get("urls", {})
            names = [o.get("name") for o in f.get("operators", []) if o.get("name")] + feed_ops.get(f["id"], [])
            if not names and len(op_names) == 1:
                names = [n for n in op_names.values() if n]
            out.append(AtlasFeed(f["id"], urls.get("static_current"), list(urls.get("static_historic", [])), names))
    return out


def load(data_dir: Path, fallback_files: list[str]) -> list[AtlasFeed]:
    feeds_dir = _sync_repo(data_dir / "transitland-atlas")
    records = []
    if feeds_dir is not None:
        for fn in sorted(feeds_dir.glob("*.dmfr.json")):
            try:
                records.append(json.loads(fn.read_text()))
            except (OSError, json.JSONDecodeError):
                continue
    else:
        for name in fallback_files:
            try:
                r = requests.get(RAW + name, timeout=60)
                r.raise_for_status()
                records.append(r.json())
            except requests.RequestException as e:
                log.warning("atlas file %s unavailable: %s", name, e)
    feeds = _parse(records)
    log.info("transitland atlas: %d GTFS feeds", len(feeds))
    return feeds


def match(feeds: list[AtlasFeed], urls: list[str | None], provider: str) -> list[tuple[AtlasFeed, bool]]:
    """Atlas records for a catalog feed as (feed, matched_by_url): records
    sharing a URL first, else records whose operator has the same name."""
    want = {norm_url(u) for u in urls if u}
    by_url = [(f, True) for f in feeds if f.url and f.all_urls() & want]
    if by_url:
        return by_url
    name = norm_name(provider)
    return [(f, False) for f in feeds if f.url and any(norm_name(o) == name for o in f.operators)]
