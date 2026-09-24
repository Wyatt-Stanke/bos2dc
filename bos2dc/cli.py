"""Command line interface.

    bos2dc fetch            discover + download GTFS feeds (Mobility Database + Transitland Atlas)
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
    p.add_argument("--max-walk", type=float, help="km: longest walk allowed (default: gap radius, raised automatically if nothing connects)")
    p.add_argument("--no-auto-walk", action="store_true", help="fail instead of raising the walk limit when nothing connects")
    p.add_argument("--city-penalty", type=float, default=1.0, help="minutes added per mile of city-density riding (2-4 stops/mi)")
    p.add_argument("--express-penalty", type=float, default=3.0, help="minutes added per mile of express riding (<2 stops/mi)")
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
    f.add_argument("--no-atlas", action="store_true", help="skip the Transitland Atlas (stale-feed refresh and extra feeds)")
    r = sub.add_parser("route", help="compute route from downloaded feeds")
    _add_route_args(r)
    ch = sub.add_parser("chain", help="evaluate a given sequence of routes (stop density, frequency, transfers)")
    _add_route_args(ch)
    ch.add_argument("routes", nargs="+", help="route names in order; AGENCY_REGEX:ROUTE to pin an agency")
    ch.add_argument("--agency", help="default agency regex for route names (e.g. 'NJ TRANSIT')")
    ch.add_argument("--from", dest="start", default="south-station", help="lat,lon or south-station / union-station")
    ch.add_argument("--to", dest="end", default="union-station", help="lat,lon or south-station / union-station")
    ch.add_argument("--transfer-walk", type=float, default=800.0, help="metres allowed between consecutive routes")
    ch.add_argument("--compare", action="store_true", help="also show the minimum-express route between the same points")
    both = sub.add_parser("run", help="fetch if needed, then route")
    _add_route_args(both)
    both.add_argument("--refetch", action="store_true")
    both.add_argument("--no-atlas", action="store_true", help="skip the Transitland Atlas when fetching")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    today = dt.date.today()

    if args.cmd == "fetch" or (args.cmd == "run" and (args.refetch or not (args.data_dir / "manifest.json").exists())):
        feeds = catalog.fetch(args.data_dir, today, buffer_km=getattr(args, "buffer_km", config.CORRIDOR_BUFFER_KM),
                              use_atlas=not args.no_atlas)
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
        cost=CostParams(wait_factor=args.wait_factor, board_penalty_s=args.board_penalty * 60, walk_factor=args.walk_factor,
                        city_penalty_s_per_mile=args.city_penalty * 60, express_penalty_s_per_mile=args.express_penalty * 60),
        max_walk_m=args.max_walk * 1000 if args.max_walk else None,
        auto_walk=not args.no_auto_walk,
        exclude_agency=args.exclude_agency,
        exclude_feed=args.exclude_feed,
        max_stale_days=args.max_stale_days,
        workers=args.workers,
        use_flex=args.flex,
        flex_files=args.flex_zones,
    )
    if args.cmd == "chain":
        return _chain(args, opts)
    res = pipeline.run(opts)
    print(f"Representative day: {WEEKDAYS[date.weekday()].title()} {date.isoformat()}")
    print(report.text_report(res))
    for t in args.itinerary:
        for i in range(len(res.routes)):
            print()
            print(report.itinerary_text(res, i, _clock(t)))
    report.write_outputs(res, args.out)
    print(f"\nwrote {args.out / 'route.json'} and {args.out / 'route.geojson'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())


def _place(text: str) -> config.Place:
    named = {"south-station": config.SOUTH_STATION, "union-station": config.UNION_STATION}
    if text in named:
        return named[text]
    lat, lon = (float(x) for x in text.split(","))
    return config.Place(text, lat, lon)


def _chain(args, opts) -> int:
    from . import pipeline
    from .chain import ChainStep, evaluate_chain, format_route
    from .geo import project
    from .search import CostParams, Searcher

    start, end = _place(args.start), _place(args.end)
    g, _ = pipeline.build(opts, start, end)
    steps = [ChainStep.parse(t, args.agency) for t in args.routes]
    oxy = project([start.lat], [start.lon])[0]
    dxy = project([end.lat], [end.lon])[0]
    r, problems = evaluate_chain(g, steps, oxy, dxy, opts.cost, transfer_walk_m=args.transfer_walk)
    print(f"Chain {' → '.join(s.route for s in steps)} from {start.name} to {end.name}")
    if problems:
        print("  " + "; ".join(problems))
    else:
        print("\n".join(format_route(g, r, steps)))
    if args.compare:
        s = Searcher(g, CostParams(city_penalty_s_per_mile=60, express_penalty_s_per_mile=6000), opts.max_walk_m)
        best = s.best_route(float(s.levels[0]))
        print(f"\nMinimum-express route between the same points (walks ≤ {s.max_walk_m:.0f} m)")
        print("\n".join(format_route(g, best)) if best else "  none")
    return 0 if not problems else 1
