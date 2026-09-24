"""Human-readable, JSON and GeoJSON output."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .graph import effective_frequency
from .itinerary import fmt_clock, fmt_dur, serving_options
from .pipeline import Result, route_feeds, summarize_profile
from .search import Route


def _hw(seconds: float | None) -> str:
    return "—" if seconds is None or not np.isfinite(seconds) else fmt_dur(seconds)


def _ride_stats(res: Result, u: int, v: int, cls: int) -> dict:
    """Exact departure statistics of a pooled leg (union over its routes)."""
    g, p = res.graph, res.graph.params
    opts = serving_options(g, u, v, cls)
    deps = np.unique(np.concatenate([o.dep for o in opts])) if opts else np.zeros(0)
    in_win = deps[(deps >= p.window_start) & (deps < p.window_end)]
    eff = float(effective_frequency(deps[:, None], p.window_start, p.window_end)[0]) if len(deps) else 0.0
    by_route: dict[tuple[str, str], int] = {}
    for o in opts:
        n = int(((o.dep >= p.window_start) & (o.dep < p.window_end)).sum())
        by_route[(o.agency, o.route_name)] = by_route.get((o.agency, o.route_name), 0) + n
    services = sorted(((n, a, r) for (a, r), n in by_route.items() if n), reverse=True)
    return {
        "departures_in_window": int(len(in_win)),
        "departures_all_day": int(len(deps)),
        "first": int(deps.min()) if len(deps) else None,
        "last": int(deps.max()) if len(deps) else None,
        "effective_headway_s": p.window_s / eff if eff else None,
        "services": [{"agency": a, "route": r, "departures": n} for n, a, r in services[:4]],
    }


def describe_route(res: Result, r: Route) -> list[dict]:
    g = res.graph
    out = []
    for leg in r.legs:
        a, b = g.nodes.iloc[leg.u], g.nodes.iloc[leg.v]
        item = {
            "kind": leg.kind,
            "from": str(a["name"]),
            "to": str(b["name"]),
            "from_latlon": [float(a["lat"]), float(a["lon"])],
            "to_latlon": [float(b["lat"]), float(b["lon"])],
            "time_s": round(leg.time_s),
        }
        if leg.kind == "walk":
            item["distance_m"] = round(leg.dist_m)
        elif leg.kind == "flex":
            z = g.zones[leg.zone]
            item.update(
                effective_headway_s=leg.headway_s,
                miles=round(sum(leg.miles), 1),
                zone={"name": z.name, "agency": z.agency, "info_url": z.info_url,
                      "hours": [[fmt_clock(s0), fmt_clock(s1)] for s0, s1, _ in z.windows(g.day)]},
                services=[{"agency": z.agency, "route": f"{z.name} (on demand)"}],
            )
        else:
            item.update(_ride_stats(res, leg.u, leg.v, int(np.argmax(leg.miles))))
            item["class_miles"] = {n: round(m, 1) for n, m in zip(g.class_names, leg.miles)}
        item["bottleneck"] = leg.kind != "walk" and abs(leg.freq - r.bottleneck_freq) < 0.005
        out.append(item)
    return out


def _class_line(res: Result, r: Route) -> str:
    m = r.miles
    total = m.sum()
    parts = [f"{n} {v:.0f} mi ({100 * v / total:.0f}%)" for n, v in zip((*res.graph.class_names, "flex"), m) if v > 0.05]
    return " · ".join(parts)


def _by_road(res: Result) -> bool:
    return len(res.graph.class_names) == 4


# Short column headers for the summary table.
_SHORT = {"local": "local", "city": "city", "express": "expr", "1 lane": "1-ln", "2 lanes": "2-ln", "3+ lanes": "3+ln",
          "controlled access": "CA"}


def text_report(res: Result) -> str:
    p = res.graph.params
    lines = [
        f"Frequency window {fmt_clock(p.window_start)}–{fmt_clock(p.window_end)}. Effective headway = 2 × the average wait of",
        "someone arriving at a random moment in the window (the plain headway for evenly spaced service; much",
        "longer for bunched or peak-only service).",
        ("Road type (OpenStreetMap): lanes in the bus's direction of travel, or controlled access (freeways and ramps)."
         if _by_road(res) else "Stop density: local ≥ 4 stops/mi, city ≥ 2, else express."),
    ]
    if res.walk_note:
        lines += ["", "NOTE: " + res.walk_note]
    for i, r in enumerate(res.routes):
        lines += ["", "=" * 100, r.label]
        lines += _summary(res, r)
        lines += _profile_block(res, i, limit=None if i == 0 else 4)
        lines.append("")
        lines += _legs_table(res, r)
        lines += _data_notes(res, r)
    lines += ["", "=" * 100, "SUMMARY"]
    heads = [_SHORT.get(n, n[:5]) for n in (*res.graph.class_names, "flex")]
    lines.append(f"  {'route':<40} {'weakest':>8} {'boards':>6} {'longest':>8} " + " ".join(f"{h:>5}" for h in heads) + f" {'exp.time':>9} {'fastest':>8}")
    lines.append(f"  {'':<40} {'headway':>8} {'':>6} {'walk':>8} " + " ".join(f"{'mi':>5}" for _ in heads) + f" {'':>9} {'(tt)':>8}")
    for i, r in enumerate(res.routes):
        prof = summarize_profile(res.profiles[i])
        m = r.miles
        fastest = fmt_dur(prof["min_duration_s"]) if prof["connections"] else "—"
        lines.append(f"  {r.label.split(':')[0][:40]:<40} {_hw(r.bottleneck_headway_s):>8} {r.boardings:>6} "
                     f"{r.longest_walk_m / 1000:>6.1f}km " + " ".join(f"{x:>5.0f}" for x in m) +
                     f" {fmt_dur(r.expected_time_s):>9} {fastest:>8}")
    return "\n".join(lines)


def _summary(res: Result, r: Route) -> list[str]:
    return [
        f"  weakest link: effective headway {_hw(r.bottleneck_headway_s)} · longest walk {r.longest_walk_m / 1000:.1f} km",
        f"  {r.boardings} boardings · in-vehicle {fmt_dur(r.in_vehicle_s)} · walking {fmt_dur(r.walk_s)}"
        f" · expected waiting {fmt_dur(r.expected_wait_s)} · frequency-based trip time {fmt_dur(r.expected_time_s)}",
        f"  {'road type' if _by_road(res) else 'stop density'}: {_class_line(res, r)}",
    ]


def _legs_table(res: Result, r: Route) -> list[str]:
    rows = []
    for k, leg in enumerate(describe_route(res, r), start=1):
        if leg["kind"] == "walk":
            rows.append(f"  {k:>2}. walk {leg['distance_m']} m ({fmt_dur(leg['time_s'])}) → {leg['to']}")
            continue
        many = len(leg["services"]) > 1
        svc = ", ".join(f"{s['agency']} {s['route']}" + (f" ({s['departures']})" if many else "") for s in leg["services"])
        flag = "   ◀ weakest link" if leg["bottleneck"] else ""
        rows.append(f"  {k:>2}. {svc}")
        rows.append(f"      {leg['from']} → {leg['to']}")
        if leg["kind"] == "flex":
            hours = ", ".join(f"{a}–{b}" for a, b in leg["zone"]["hours"])
            rows.append(f"      on demand {hours}, book by app or phone · eff. headway {_hw(leg['effective_headway_s'])}"
                        f" (assumed response) · ≈{leg['miles']} mi, ride ≈{fmt_dur(leg['time_s'])}{flag}")
            if leg["zone"]["info_url"]:
                rows.append(f"      {leg['zone']['info_url']}")
        else:
            span = f"{fmt_clock(leg['first'])}–{fmt_clock(leg['last'])}" if leg["first"] is not None else ""
            cm = " / ".join(f"{n} {m}" for n, m in leg["class_miles"].items() if m >= 0.05)
            rows.append(f"      {leg['departures_in_window']} departures in window ({leg['departures_all_day']} all day, {span})"
                        f" · eff. headway {_hw(leg['effective_headway_s'])} · ride {fmt_dur(leg['time_s'])} · {cm} mi{flag}")
    return rows


def _profile_block(res: Result, i: int, limit: int | None = None) -> list[str]:
    prof = res.profiles[i]
    s = summarize_profile(prof)
    if not s["connections"]:
        return ["  timetable check: no complete connection found"]
    lines = [
        f"  timetable check (same weekday timetable every day): {s['connections']} distinct connections per day;"
        f" door-to-door {fmt_dur(s['min_duration_s'])} fastest, {fmt_dur(s['median_duration_s'])} median"
    ]
    shown = prof if limit is None else prof[:limit]
    for dep, arr in shown:
        lines.append(f"    leave South Station by {fmt_clock(dep)} → Union Station {fmt_clock(arr)}  ({fmt_dur(arr - dep)})")
    if limit is not None and len(prof) > limit:
        lines.append(f"    … {len(prof) - limit} more")
    return lines


def _data_notes(res: Result, r: Route) -> list[str]:
    lines = []
    for fid in sorted(route_feeds(res.graph, r)):
        c = res.manifest.get(fid)
        f = res.graph.feeds[fid]
        notes = list(f.notes) + (list(c.notes) if c else [])
        if c and c.stale:
            lines.append(f"  ⚠ {c.provider} ({fid}): newest data ends {c.service_end}; timetable of {f.service_date} used")
        elif notes:
            lines.append(f"  · {f.provider} ({fid}): {'; '.join(notes)}")
    zones = {res.graph.zones[l.zone].name for l in r.legs if l.kind == "flex"}
    for z in sorted(zones):
        lines.append(f"  · {z}: zone polygon and hours from the agency's map layer; frequency assumes a "
                     f"{fmt_dur(res.graph.params.flex_headway_s)} equivalent headway (--flex-headway)")
    if lines:
        lines = ["  data notes:"] + lines
    return lines


def itinerary_text(res: Result, i: int, t0: int) -> str:
    arr, steps = res.simulators[i].run(t0, detail=True)
    lines = [f"{res.routes[i].label.split(':')[0]} — leaving South Station at {fmt_clock(t0)}:"]
    for st in steps:
        if st.kind in ("ride", "flex", "walk"):
            lines.append(f"  {fmt_clock(st.start):>13}–{fmt_clock(st.end):<13} {st.label}")
        elif st.end - st.start >= 60:
            lines.append(f"  {'':>27} wait {fmt_dur(st.end - st.start)}")
    lines.append("  no connection" if arr is None else f"  arrive Union Station {fmt_clock(arr)} ({fmt_dur(arr - t0)} door to door)")
    return "\n".join(lines)


def to_json(res: Result) -> dict:
    p = res.graph.params
    return {
        "window": [p.window_start, p.window_end],
        "max_bottleneck_effective_departures": res.best_threshold,
        "max_walk_m": res.max_walk_m,
        "walk_note": res.walk_note,
        "routes": [
            {
                "label": r.label,
                "bottleneck_effective_headway_s": r.bottleneck_headway_s,
                "boardings": r.boardings,
                "in_vehicle_s": r.in_vehicle_s,
                "walk_s": r.walk_s,
                "longest_walk_m": r.longest_walk_m,
                "expected_wait_s": r.expected_wait_s,
                "miles": dict(zip((*res.graph.class_names, "flex"), (round(x, 1) for x in r.miles))),
                "legs": describe_route(res, r),
                "connections": [[d, a] for d, a in res.profiles[i]],
                "profile": summarize_profile(res.profiles[i]),
                "feeds": sorted(route_feeds(res.graph, r)),
            }
            for i, r in enumerate(res.routes)
        ],
    }


def to_geojson(res: Result) -> dict:
    features = []
    for i, r in enumerate(res.routes):
        for leg in describe_route(res, r):
            coords = [[leg["from_latlon"][1], leg["from_latlon"][0]], [leg["to_latlon"][1], leg["to_latlon"][0]]]
            props = {"route_rank": i, "route_label": r.label, "kind": leg["kind"], "from": leg["from"], "to": leg["to"]}
            if leg["kind"] != "walk":
                props["services"] = ", ".join(f"{s['agency']} {s['route']}" for s in leg["services"])
                props["effective_headway_s"] = leg["effective_headway_s"]
            features.append({"type": "Feature", "geometry": {"type": "LineString", "coordinates": coords}, "properties": props})
    return {"type": "FeatureCollection", "features": features}


def write_outputs(res: Result, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "route.json").write_text(json.dumps(to_json(res), indent=1))
    (out_dir / "route.geojson").write_text(json.dumps(to_geojson(res)))
