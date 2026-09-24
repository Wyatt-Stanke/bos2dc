import datetime as dt
import zipfile

import numpy as np
import pytest

from bos2dc.config import Place
from bos2dc.graph import GraphParams, attach_places, build_graph
from bos2dc.gtfs import compile_feed
from bos2dc.itinerary import Simulator
from bos2dc.search import CostParams, Searcher

DAY = dt.date(2026, 9, 30)  # a Wednesday


def hms(sec):
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def make_feed(path, stops, routes, calendar_days="1,1,1,1,1,0,0", freq_rows=(), shapes=None):
    """routes: {route_id: [(first_dep_s, [(stop_id, offset_s), ...]), ...]}
    shapes: {route_id: [(lat, lon), ...]} drawn for every trip of the route."""
    st_rows, trip_rows = [], []
    shapes = shapes or {}
    for rid, trips in routes.items():
        for k, (t0, seq) in enumerate(trips):
            tid = f"{rid}_{k}"
            trip_rows.append(f"{rid},WK,{tid},to end,{'S' + rid if rid in shapes else ''}")
            for i, (sid, off) in enumerate(seq):
                st_rows.append(f"{tid},{hms(t0 + off)},{hms(t0 + off)},{sid},{i + 1}")
    files = {
        "agency.txt": "agency_id,agency_name,agency_url,agency_timezone\nA,Test Agency " + path.stem + ",http://x,America/New_York\n",
        "stops.txt": "stop_id,stop_name,stop_lat,stop_lon\n" + "".join(f"{s},{n},{la},{lo}\n" for s, n, la, lo in stops),
        "routes.txt": "route_id,agency_id,route_short_name,route_long_name,route_type\n" + "".join(f"{r},A,{r},,3\n" for r in routes),
        "trips.txt": "route_id,service_id,trip_id,trip_headsign,shape_id\n" + "\n".join(trip_rows) + "\n",
        "stop_times.txt": "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n" + "\n".join(st_rows) + "\n",
        "calendar.txt": "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date\n"
                        f"WK,{calendar_days},20260101,20261231\n",
    }
    if shapes:
        files["shapes.txt"] = "shape_id,shape_pt_lat,shape_pt_lon,shape_pt_sequence\n" + "".join(
            f"S{rid},{la},{lo},{i}\n" for rid, pts in shapes.items() for i, (la, lo) in enumerate(pts))
    if freq_rows:
        files["frequencies.txt"] = "trip_id,start_time,end_time,headway_secs\n" + "".join(f"{t},{a},{b},{h}\n" for t, a, b, h in freq_rows)
    with zipfile.ZipFile(path, "w") as z:
        for name, body in files.items():
            z.writestr(name, body)
    return path


def every(start_h, end_h, minutes, seq):
    return [(t, seq) for t in range(start_h * 3600, end_h * 3600, minutes * 60)]


ORIGIN = Place("O", 40.000, -75.000)
DEST = Place("D", 40.300, -75.000)


@pytest.fixture
def network(tmp_path):
    # Feed 1 leaves the origin two ways: a fast daily bus to B, or a
    # quarter-hourly bus to C.
    f1 = make_feed(tmp_path / "f1.zip",
                   [("A", "Alpha", 40.000, -75.000), ("B", "Bravo", 40.150, -74.950), ("C", "Charlie", 40.150, -75.050)],
                   {"FAST": [(8 * 3600, [("A", 0), ("B", 600)])],
                    "FREQ": every(6, 22, 15, [("A", 0), ("C", 1800)])})
    # Feed 2 (another agency) continues from B or C to the destination. From
    # C two hourly routes share the corridor (pooled: 30 min).
    f2 = make_feed(tmp_path / "f2.zip",
                   [("B2", "Bravo2", 40.1501, -74.950), ("C2", "Charlie2", 40.1501, -75.050), ("D", "Delta", 40.300, -75.000)],
                   {"BX": every(6, 22, 10, [("B2", 0), ("D", 600)]),
                    "C1": every(6, 22, 60, [("C2", 0), ("D", 1200)]),
                    "C2": [(t + 1800, s) for t, s in every(6, 22, 60, [("C2", 0), ("D", 1500)])]})
    feeds = [compile_feed(f1, "f1", "one", DAY), compile_feed(f2, "f2", "two", DAY)]
    g = build_graph(feeds, GraphParams())
    return attach_places(g, ORIGIN, DEST)


def test_compile_patterns(network):
    f1 = network.feeds["f1"]
    names = sorted(p.route_name for p in f1.patterns)
    assert names == ["FAST", "FREQ"]
    freq = next(p for p in f1.patterns if p.route_name == "FREQ")
    assert freq.dep.shape == (64, 2)


def test_common_lines_are_pooled(network):
    g = network
    s = g.nodes.index[g.nodes["stop_id"] == "C2"][0]
    d = g.nodes.index[g.nodes["stop_id"] == "D"][0]
    k = np.flatnonzero((g.ride_u == s) & (g.ride_v == d))
    assert len(k) == 1
    assert g.ride_freq[k[0]] == pytest.approx(32)  # two interleaved hourly routes over 16 h


def test_bottleneck_prefers_frequent_chain(network):
    s = Searcher(network, CostParams())
    best = s.max_bottleneck()
    assert best == pytest.approx(32)  # limited by the pooled C -> D leg, not the daily bus
    r = s.best_route(best)
    stops = [network.nodes.at[i, "stop_id"] for i in r.nodes]
    assert "C" in stops and "B" not in stops
    assert r.bottleneck_freq == pytest.approx(32)


def test_frontier_includes_faster_weaker_option(network):
    s = Searcher(network, CostParams(wait_factor=0.0, board_penalty_s=0))
    frontier = s.frontier(s.max_bottleneck())
    assert frontier[0].bottleneck_freq == pytest.approx(32)
    # With waiting ignored the daily bus is quicker, so it shows up as a
    # weaker-bottleneck alternative.
    assert any(r.bottleneck_freq == pytest.approx(1) for r in frontier[1:])


def test_timetable_profile(network):
    s = Searcher(network)
    r = s.best_route(s.max_bottleneck())
    sim = Simulator(network, r)
    arr, steps = sim.run(7 * 3600, detail=True)
    assert arr is not None and arr > 7 * 3600
    rides = [st for st in steps if st.kind == "ride"]
    assert len(rides) == 2
    prof = sim.profile(step_s=60)
    assert len(prof) >= 30
    assert all(a > d for d, a in prof)


def test_frequencies_txt_expansion(tmp_path):
    f = make_feed(tmp_path / "fq.zip",
                  [("A", "A", 40.0, -75.0), ("B", "B", 40.01, -75.0)],
                  {"R": [(0, [("A", 0), ("B", 300)])]},
                  freq_rows=[("R_0", "07:00:00", "09:00:00", 600)])
    feed = compile_feed(f, "fq", "fq", DAY)
    assert feed.patterns[0].dep.shape[0] == 12
    assert feed.patterns[0].dep[0, 0] == 7 * 3600


def test_stale_feed_maps_to_same_weekday(tmp_path):
    f = make_feed(tmp_path / "old.zip",
                  [("A", "A", 40.0, -75.0), ("B", "B", 40.01, -75.0)],
                  {"R": every(6, 8, 30, [("A", 0), ("B", 300)])})
    # Rewrite the calendar to end in 2025.
    with zipfile.ZipFile(f) as z:
        files = {n: z.read(n) for n in z.namelist()}
    files["calendar.txt"] = files["calendar.txt"].replace(b"20261231", b"20250601").replace(b"20260101", b"20250101")
    with zipfile.ZipFile(f, "w") as z:
        for n, b in files.items():
            z.writestr(n, b)
    feed = compile_feed(f, "old", "old", DAY)
    assert feed.service_date.weekday() == DAY.weekday()
    assert feed.service_date <= dt.date(2025, 6, 1)
    assert feed.notes


def test_effective_frequency():
    from bos2dc.graph import effective_frequency
    w0, w1 = 6 * 3600, 22 * 3600
    hourly = np.arange(w0, w1, 3600)[:, None]
    assert effective_frequency(hourly, w0, w1)[0] == pytest.approx(16)
    daily = np.array([[9 * 3600]])
    assert effective_frequency(daily, w0, w1)[0] == pytest.approx(1)
    peak = np.array([6, 6.5, 7, 7.5])[:, None] * 3600
    assert effective_frequency(peak, w0, w1)[0] == pytest.approx(256 / 211, rel=1e-3)
    outside = np.array([[23 * 3600]])
    assert effective_frequency(outside, w0, w1)[0] == 0


def _square_zone(name, lat0, lon0, size, hours=(6 * 3600, 18 * 3600)):
    from bos2dc.flex import FlexZone
    ring = np.array([[lon0, lat0], [lon0 + size, lat0], [lon0 + size, lat0 + size], [lon0, lat0 + size], [lon0, lat0]])
    return FlexZone(name, name, "Agency", [ring], {d: [(hours[0], hours[1], None)] for d in range(5)})


def test_flex_zone_bridges_gap(tmp_path):
    from bos2dc.graph import add_flex
    # Two agencies 5 km apart; a Flex zone covers the first agency's
    # terminal and ends 1 km short of the second agency's first stop.
    f1 = make_feed(tmp_path / "a.zip", [("A1", "A1", 40.00, -75.00), ("A2", "A2", 40.05, -75.00)],
                   {"R": every(6, 22, 30, [("A1", 0), ("A2", 600)])})
    f2 = make_feed(tmp_path / "b.zip", [("B1", "B1", 40.10, -75.00), ("B2", "B2", 40.30, -75.00)],
                   {"S": every(6, 22, 30, [("B1", 0), ("B2", 900)])})
    feeds = [compile_feed(f1, "a", "a", DAY), compile_feed(f2, "b", "b", DAY)]
    params = GraphParams(gap_radius_m=1500)
    g = attach_places(build_graph(feeds, params), ORIGIN, DEST)
    assert Searcher(g).max_bottleneck() == 0
    zone = _square_zone("Z", 40.04, -75.01, 0.05)  # lat 40.04-40.09: ~1.1 km short of B1
    g = attach_places(build_graph(feeds, params), ORIGIN, DEST)
    g = add_flex(g, [zone, _square_zone("City Paratransit", 40.0, -75.1, 0.5)], DAY)
    assert len(g.zones) == 1  # paratransit skipped
    s = Searcher(g)
    best = s.max_bottleneck()
    assert best > 0
    r = s.best_route(best)
    kinds = [l.kind for l in r.legs]
    assert "flex" in kinds
    assert kinds[kinds.index("flex") + 1] == "walk"  # zone edge -> walk to B1
    # 06:00-18:00 hourly-equivalent service in a 06:00-22:00 window.
    assert r.bottleneck_freq == pytest.approx(256 / (12 + 16), abs=0.01)
    sim = Simulator(g, r)
    arr, steps = sim.run(7 * 3600, detail=True)
    assert arr is not None and any(st.kind == "flex" for st in steps)
    # After the zone closes the trip rolls to the next morning.
    late, _ = sim.run(19 * 3600)
    assert late > 24 * 3600


def test_parse_hours():
    from bos2dc.flex import parse_hours
    assert parse_hours("6:00am-6:30pm") == [(6 * 3600, 18 * 3600 + 1800, None)]
    assert parse_hours("6:45am-5:12pm: every 90 min") == [(6 * 3600 + 2700, 17 * 3600 + 720, 5400)]
    assert parse_hours("NO SERVICE") == []


def test_segment_classes():
    from bos2dc.graph import CITY, EXPRESS, LOCAL, segment_classes
    # 10 hops of 0.2 mi (5/mi), 5 of 0.4 mi (2.5/mi), then 3 highway hops of 5 mi.
    hops = np.array([0.2] * 10 + [0.4] * 5 + [5.0] * 3)
    cls = segment_classes(hops)
    assert (cls[:8] == LOCAL).all()
    assert (cls[11:15] == CITY).all()
    assert (cls[15:] == EXPRESS).all()


def test_local_preference_picks_local_run(tmp_path):
    # Two parallel routes A -> D: a local with a stop every 0.1 mi and an
    # express with no intermediate stops. Same frequency; the express is faster.
    lat = [40.0 + 0.00145 * k for k in range(21)]  # ~0.1 mi apart, 2 mi total
    stops = [(f"L{k}", f"L{k}", la, -75.0) for k, la in enumerate(lat)]
    local_seq = [(f"L{k}", 90 * k) for k in range(21)]
    express_seq = [("L0", 0), ("L20", 600)]
    f = make_feed(tmp_path / "loc.zip", stops, {"LOC": every(6, 22, 30, local_seq), "EXP": every(6, 22, 30, express_seq)})
    feed = compile_feed(f, "loc", "loc", DAY)
    g = attach_places(build_graph([feed], GraphParams()), Place("O", lat[0], -75.0), Place("D", lat[-1], -75.0))
    fast = Searcher(g, CostParams(city_penalty_s_per_mile=0, express_penalty_s_per_mile=0))
    r = fast.best_route(fast.max_bottleneck())
    assert [l.miles for l in r.legs if l.kind == "ride"][0][2] > 1.5  # took the express
    local = Searcher(g, CostParams.most_local())
    r = local.best_route(local.max_bottleneck())
    ride = [l for l in r.legs if l.kind == "ride"]
    assert len(ride) == 1 and ride[0].miles[0] > 1.5  # took the local
    sim = Simulator(g, r)
    arr, steps = sim.run(7 * 3600, detail=True)
    assert "LOC" in [st for st in steps if st.kind == "ride"][0].label


def test_required_walk(tmp_path):
    # Two networks 3 km apart: unreachable at the default 2.5 km gap radius,
    # connected once walks up to ~3 km are allowed.
    f1 = make_feed(tmp_path / "a.zip", [("A1", "A1", 40.000, -75.0), ("A2", "A2", 40.050, -75.0)],
                   {"R": every(6, 22, 30, [("A1", 0), ("A2", 600)])})
    f2 = make_feed(tmp_path / "b.zip", [("B1", "B1", 40.077, -75.0), ("B2", "B2", 40.300, -75.0)],
                   {"S": every(6, 22, 30, [("B1", 0), ("B2", 900)])})
    feeds = [compile_feed(f1, "a", "a", DAY), compile_feed(f2, "b", "b", DAY)]
    g = attach_places(build_graph(feeds, GraphParams()), ORIGIN, DEST)
    s = Searcher(g)
    assert not s.reachable(float(s.levels[0]))
    need = s.required_walk()
    assert 2900 < need < 3100
    s2 = Searcher(g, max_walk_m=need + 1)
    r = s2.best_route(s2.max_bottleneck())
    assert 2900 < r.longest_walk_m < 3100


def test_chain_evaluation(network):
    from bos2dc.chain import ChainStep, evaluate_chain
    from bos2dc.geo import project
    o = project([ORIGIN.lat], [ORIGIN.lon])[0]
    d = project([DEST.lat], [DEST.lon])[0]
    r, problems = evaluate_chain(network, [ChainStep("Agency f1", "FAST"), ChainStep(None, "BX")], o, d)
    assert not problems
    assert [l.kind for l in r.legs if l.kind != "walk"] == ["ride", "ride"]
    assert r.bottleneck_freq == pytest.approx(1)  # the daily FAST bus
    r, problems = evaluate_chain(network, [ChainStep(None, "FREQ"), ChainStep(None, "BX")], o, d, transfer_walk_m=500)
    assert r is None and problems  # C and B are ~8.5 km apart
