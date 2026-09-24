"""Command line interface.

    bos2dc fetch            discover + download GTFS feeds from the Mobility Database
    bos2dc route            compute the frequency-optimal route from downloaded feeds
    bos2dc run              fetch (if no manifest yet) then route
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import sys
from pathlib import Path

from . import catalog, config
from .graph import GraphParams
from .gtfs import WEEKDAYS
from .search import CostParams


def _clock(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 3600 + int(m) * 60


def _add_route_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--day", default="wednesday", choices=WEEKDAYS, help="weekday to analyse (default: next Wednesday)")
    p.add_argument("--date", type=dt.date.fromisoformat, help="exact service date (overrides --day)")
    p.add_argument("--window", default="06:00-22:00", help="time window in which departures are counted")
    p.add_argument("--wait-factor", type=float, default=0.5, help="expected wait per boarding as a fraction of headway")
    p.add_argument("--board-penalty", type=float, default=5.0, help="minutes added per boarding")
    p.add_argument("--walk-factor", type=float, default=1.5, help="weight of walking minutes relative to riding")
    p.add_argument("--walk-radius", type=float, default=400.0, help="metres: walk links between any two stops")
    p.add_argument("--gap-radius", type=float, default=2500.0, help="metres: walk links between different agencies")
    p.add_argument("--access-radius", type=float, default=800.0, help="metres: walk to/from the two stations")
    p.add_argument("--exclude-agency", action="append", default=[], metavar="REGEX", help="drop matching agencies/routes (repeatable)")
    p.add_argument("--exclude-feed", action="append", default=[], metavar="FEED_ID", help="drop a catalog feed (repeatable)")
    p.add_argument("--max-stale-days", type=int, help="ignore feeds whose data expired more than N days before the date")
    p.add_argument("--flex", action="store_true", help="also use demand-response (Flex) zones (off by default: not buses)")
    p.add_argument("--flex-zones", action="append", default=[], metavar="GEOJSON", help="extra Flex zones file (repeatable; see README)")
    p.add_argument("--flex-headway", type=float, default=60.0, help="minutes: equivalent headway of an on-demand zone")
    p.add_argument("--itinerary", metavar="HH:MM", action="append", default=[], help="print a timed itinerary leaving at this time")
    p.add_argument("--out", type=Path, default=Path("out"), help="directory for route.json / route.geojson")
    p.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bos2dc", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download feeds")
    f.add_argument("--buffer-km", type=float, default=config.CORRIDOR_BUFFER_KM)
    r = sub.add_parser("route", help="compute route from downloaded feeds")
    _add_route_args(r)
    both = sub.add_parser("run", help="fetch if needed, then route")
    _add_route_args(both)
    both.add_argument("--refetch", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    today = dt.date.today()

    if args.cmd == "fetch" or (args.cmd == "run" and (args.refetch or not (args.data_dir / "manifest.json").exists())):
        feeds = catalog.fetch(args.data_dir, today, buffer_km=getattr(args, "buffer_km", config.CORRIDOR_BUFFER_KM))
        print(f"{len(feeds)} feeds ready in {args.data_dir}", file=sys.stderr)
        if args.cmd == "fetch":
            return 0

    from . import pipeline, report  # heavy imports only when routing

    date = args.date or pipeline.next_weekday(today, WEEKDAYS.index(args.day))
    w0, w1 = (_clock(x) for x in args.window.split("-"))
    opts = pipeline.Options(
        data_dir=args.data_dir,
        date=date,
        graph=GraphParams(window_start=w0, window_end=w1, walk_radius_m=args.walk_radius,
                          gap_radius_m=args.gap_radius, access_radius_m=args.access_radius,
                          flex_headway_s=int(args.flex_headway * 60)),
        cost=CostParams(wait_factor=args.wait_factor, board_penalty_s=args.board_penalty * 60, walk_factor=args.walk_factor),
        exclude_agency=args.exclude_agency,
        exclude_feed=args.exclude_feed,
        max_stale_days=args.max_stale_days,
        workers=args.workers,
        use_flex=args.flex,
        flex_files=args.flex_zones,
    )
    res = pipeline.run(opts)
    print(f"Representative day: {WEEKDAYS[date.weekday()].title()} {date.isoformat()}")
    print(report.text_report(res))
    for t in args.itinerary:
        print()
        print(report.itinerary_text(res, 0, _clock(t)))
    report.write_outputs(res, args.out)
    print(f"\nwrote {args.out / 'route.json'} and {args.out / 'route.geojson'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
