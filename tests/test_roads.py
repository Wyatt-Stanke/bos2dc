import numpy as np
import pytest

from bos2dc import roads
from bos2dc.config import Place
from bos2dc.geo import project
from bos2dc.graph import GraphParams, attach_places, build_graph
from bos2dc.gtfs import compile_feed
from bos2dc.itinerary import Simulator
from bos2dc.search import CostParams, Searcher
from test_core import DAY, every, make_feed

LOCAL, TWO, THREE, CA = roads.ONE_LANE, roads.TWO_LANES, roads.THREE_PLUS, roads.CONTROLLED


def write_index(path, ways):
    """ways: [(points [(lat, lon), ...], class forward, class backward, oneway[, off grade])]"""
    xy = np.vstack([project([p[0] for p in w[0]], [p[1] for p in w[0]]) for w in ways])
    np.savez(path, xy=xy, offsets=np.r_[0, np.cumsum([len(w[0]) for w in ways])],
             cls_f=np.array([w[1] for w in ways], np.int8), cls_b=np.array([w[2] for w in ways], np.int8),
             oneway=np.array([w[3] for w in ways], np.int8), source=np.zeros(len(ways), np.int8),
             hw=np.zeros(len(ways), np.int8), way_id=np.arange(len(ways)),
             off_grade=np.array([len(w) > 4 and w[4] for w in ways], bool))
    return roads.RoadIndex.load(path)


def line(lat0, lon0, lat1, lon1, n=40):
    return list(zip(np.linspace(lat0, lat1, n), np.linspace(lon0, lon1, n)))


def match(index, pts):
    lat, lon = np.array(pts).T
    s, xy, mid, hd, length = roads.resample(lat, lon)
    cls, src = index.match(mid, hd)
    return roads._smooth(cls, length)


@pytest.fixture
def index(tmp_path):
    return write_index(tmp_path / "roads.npz", [
        (line(40.0, -75.02, 40.0, -74.98), CA, CA, 0),  # freeway, east-west
        (line(39.98, -75.0, 40.02, -75.0), LOCAL, LOCAL, 0),  # local street crossing it on an overpass
        (line(40.0018, -75.02, 40.0018, -74.98), TWO, TWO, 0),  # arterial 200 m north of the freeway
        # divided road: eastbound carriageway (3 lanes) and westbound (drawn westwards, 2 lanes) 14 m apart
        (line(40.0100, -75.02, 40.0100, -74.98), THREE, LOCAL, 1),
        (line(40.01013, -74.98, 40.01013, -75.02), TWO, LOCAL, 1),
    ])


def test_lanes_by_direction():
    assert roads.lanes_by_direction({"highway": "primary", "lanes": "4"}) == (0, 2, 2)
    assert roads.lanes_by_direction({"highway": "primary", "lanes": "3"}) == (0, 1, 1)
    assert roads.lanes_by_direction({"highway": "primary", "lanes": "3", "lanes:forward": "2", "lanes:backward": "1"}) == (0, 2, 1)
    assert roads.lanes_by_direction({"highway": "secondary", "oneway": "yes", "lanes": "3"}) == (1, 3, 1)
    assert roads.lanes_by_direction({"highway": "motorway", "lanes": "3"})[:2] == (1, 3)
    o, f, b = roads.lanes_by_direction({"highway": "residential"})
    assert o == 0 and np.isnan(f) and np.isnan(b)
    assert roads.is_controlled({"highway": "motorway_link"}) and roads.is_controlled({"highway": "trunk", "motorroad": "yes"})
    assert not roads.is_controlled({"highway": "trunk", "expressway": "yes"})
    assert list(roads.lane_class(np.array([1, 2, 3, 5]))) == [LOCAL, TWO, THREE, THREE]


def test_heading_keeps_overpass_off_the_freeway(index):
    # North along the local street, over the freeway: never matched to it.
    assert set(match(index, line(39.99, -75.0, 40.008, -75.0))) == {LOCAL}
    # Along the freeway: the arterial 200 m away is out of reach.
    assert set(match(index, line(40.0, -75.015, 40.0, -74.985))) == {CA}


def test_carriageway_by_direction(index):
    # A shape drawn between the two carriageways picks the one going its way.
    mid = 40.010065
    assert set(match(index, line(mid, -75.015, mid, -74.985))) == {THREE}
    assert set(match(index, line(mid, -74.985, mid, -75.015))) == {TWO}


def test_tunnel_under_street_does_not_capture_street_buses(tmp_path):
    idx = write_index(tmp_path / "t.npz", [
        (line(40.0, -75.02, 40.0, -74.98), CA, CA, 0, True),  # freeway tunnel ...
        (line(40.00005, -75.02, 40.00005, -74.98), LOCAL, LOCAL, 0),  # ... 6 m from the street above it
        (line(40.05, -75.02, 40.05, -74.98), CA, CA, 0, True),  # a tunnel with nothing above
    ])
    assert set(match(idx, line(40.0, -75.015, 40.0, -74.985))) == {LOCAL}
    assert set(match(idx, line(40.05, -75.015, 40.05, -74.985))) == {CA}


def test_smooth_removes_blips_and_fills_gaps():
    L = np.full(43, 20.0)
    c = np.array([0] * 20 + [1] * 3 + [0] * 20, np.int8)
    assert set(roads._smooth(c, L)) == {0}  # 60 m blip (a turn-lane pocket)
    c = np.array([0] * 15 + [1] * 13 + [0] * 15, np.int8)
    assert (roads._smooth(c, L) == c).all()  # 260 m of 2-lane road is real
    c = np.array([-1, -1, 2, 2, -1, 1, 1], np.int8)
    assert list(roads._smooth(c, np.full(7, 20.0))) == [2, 2, 2, 2, 2, 1, 1]


def test_align_stops_on_out_and_back_shape():
    x = np.r_[np.arange(0, 1001, 20), np.arange(980, -1, -20)].astype(float)
    pts = np.column_stack([x, np.zeros_like(x)])
    stops = np.array([[0, 5], [500, 5], [1000, 5], [500, -5], [0, -5]], float)
    j, off = roads.align_stops(stops, pts)
    assert list(j) == [0, 25, 50, 75, 100] and off <= 5


def test_road_locality_route_choice(tmp_path):
    # Two routes A -> B at the same frequency: FWY on the freeway (fast), LOC
    # on a 1-lane street that loops north (slower).
    idx_path = tmp_path / "roads.npz"
    write_index(idx_path, [
        (line(40.0, -75.02, 40.0, -74.98), CA, CA, 0),
        ([(40.0, -75.02), (40.01, -75.02)] + line(40.01, -75.02, 40.01, -74.98) + [(40.0, -74.98)], LOCAL, LOCAL, 0),
    ])
    stops = [("A", "A", 40.0, -75.02), ("B", "B", 40.0, -74.98)]
    shapes = {"FWY": line(40.0, -75.02, 40.0, -74.98),
              "LOC": [(40.0, -75.02), (40.01, -75.02)] + line(40.01, -75.02, 40.01, -74.98) + [(40.0, -74.98)]}
    zp = make_feed(tmp_path / "r.zip", stops, {"FWY": every(6, 22, 30, [("A", 0), ("B", 300)]),
                                               "LOC": every(6, 22, 30, [("A", 0), ("B", 900)])}, shapes=shapes)
    feed = compile_feed(zp, "r", "r", DAY)
    by_road = roads.classify_feeds([feed], {"r": str(zp)}, idx_path, tmp_path / "cache", workers=1)
    assert by_road["r"].fallback_patterns == 0
    g = attach_places(build_graph([feed], GraphParams(), roads=by_road), Place("O", 40.0, -75.02), Place("D", 40.0, -74.98))
    assert g.class_names == roads.ROAD_CLASS_NAMES
    fast = Searcher(g, CostParams(road_penalties_s_per_mile=(0, 0, 0, 0)))
    leg = [l for l in fast.best_route(fast.max_bottleneck()).legs if l.kind == "ride"][0]
    assert leg.miles[CA] > 2.0 and leg.miles[LOCAL] < 0.1  # ~2.1 mi of freeway
    local = Searcher(g, CostParams.most_local())
    r = local.best_route(local.max_bottleneck())
    leg = [l for l in r.legs if l.kind == "ride"][0]
    assert leg.miles[LOCAL] > 3.0 and leg.miles[CA] < 0.1  # the 1.2 km detour north and back
    _, steps = Simulator(g, r).run(7 * 3600, detail=True)
    assert "LOC" in [st for st in steps if st.kind == "ride"][0].label
