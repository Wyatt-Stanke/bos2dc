"""Route search on the frequency graph.

Objective, in priority order:

0. **Connectivity with the shortest possible longest walk.** Walks between
   agencies are normally limited to `gap_radius`. If that leaves Union
   Station unreachable, the smallest walking distance that connects the two
   networks is found (binary search over walk lengths) and walks up to that
   length are allowed.

1. **Bottleneck frequency** (maximise). A chain of buses is only as usable as
   its least frequent leg: that leg sets how many real departure
   opportunities per day the whole trip has and how long you are stranded if
   anything upstream runs late. This is the *widest path* (max-min) problem;
   it is solved exactly by binary search over the distinct leg frequencies
   with a reachability test at each threshold.

2. **Generalised cost** (minimise) among routes that achieve that
   bottleneck: in-vehicle time + `wait_factor` x effective headway per
   boarding + a boarding penalty + weighted walking + a per-mile penalty for
   each locality class: by road type (more lanes, then controlled access,
   cost more) or by stop density (city, then express). With the penalties
   at zero and wait_factor = 0.5 this is the expected door-to-door time of a
   traveller who shows up without consulting timetables.

A lexicographic (bottleneck, cost) label is not order-preserving under edge
extension, so a single Dijkstra pass cannot optimise both; the two-phase
threshold approach is exact. Running phase 2 for a ladder of thresholds
gives the Pareto frontier of (bottleneck headway, cost).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, dijkstra

from .graph import METRES_PER_MILE, Graph

HEADWAY_LADDER_MIN = (10, 15, 20, 30, 40, 45, 60, 75, 90, 120, 150, 180, 240, 300, 360, 480, 720)


@dataclass
class CostParams:
    wait_factor: float = 0.5
    board_penalty_s: float = 300.0
    walk_factor: float = 1.5  # walking minutes weigh more than riding ones
    long_walk_m: float = 1000.0  # walks beyond this are weighted by long_walk_factor
    long_walk_factor: float = 3.0
    # Stop-density preference, seconds per mile ridden in each class. The
    # express-over-city step is deliberately larger than city-over-local.
    city_penalty_s_per_mile: float = 60.0
    express_penalty_s_per_mile: float = 180.0
    # Road-type preference, seconds per mile on roads with 1, 2, 3+ lanes in
    # the bus's direction and on controlled-access roads. As with stop
    # density, the step down to the least local class is the largest.
    road_penalties_s_per_mile: tuple[float, ...] = (0.0, 60.0, 120.0, 240.0)
    flex_penalty_s_per_mile: float = 60.0  # on-demand zones count like city running

    @classmethod
    def most_local(cls, **kw) -> "CostParams":
        """Localness dominates: 5 and 10 min per mile on 2- and 3+-lane roads,
        20 on controlled access (stop density: 5 per city mile, 15 per express)."""
        return cls(city_penalty_s_per_mile=300.0, express_penalty_s_per_mile=900.0,
                   road_penalties_s_per_mile=(0.0, 300.0, 600.0, 1200.0), flex_penalty_s_per_mile=300.0, **kw)

    def class_penalties(self, class_names: tuple) -> np.ndarray:
        from .roads import ROAD_CLASS_NAMES

        if tuple(class_names) == ROAD_CLASS_NAMES:
            return np.asarray(self.road_penalties_s_per_mile, float)
        return np.array([0.0, self.city_penalty_s_per_mile, self.express_penalty_s_per_mile])


@dataclass
class Leg:
    kind: str  # "ride" | "flex" | "walk"
    u: int
    v: int
    time_s: float
    freq: float = 0.0  # effective departures per window (pooled for rides, equivalent for flex)
    headway_s: float = 0.0
    zone: int = -1  # index into Graph.zones for flex legs
    miles: tuple[float, ...] = (0.0, 0.0, 0.0)  # per locality class (Graph.class_names)
    dist_m: float = 0.0  # walks


@dataclass
class Route:
    threshold: float
    nodes: list[int]
    legs: list[Leg]
    cost_s: float
    bottleneck_freq: float
    in_vehicle_s: float = 0.0
    walk_s: float = 0.0
    expected_wait_s: float = 0.0
    boardings: int = 0
    window_s: int = 16 * 3600
    label: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def bottleneck_headway_s(self) -> float:
        return self.window_s / self.bottleneck_freq if self.bottleneck_freq else float("inf")

    @property
    def expected_time_s(self) -> float:
        return self.in_vehicle_s + self.walk_s + self.expected_wait_s

    @property
    def miles(self) -> np.ndarray:
        """Miles by locality class (Graph.class_names), then flex."""
        K = max((len(l.miles) for l in self.legs if l.kind != "walk"), default=3)
        out = np.zeros(K + 1)
        for l in self.legs:
            if l.kind == "ride":
                out[:K] += l.miles
            elif l.kind == "flex":
                out[K] += sum(l.miles)
        return out

    @property
    def longest_walk_m(self) -> float:
        return max((l.dist_m for l in self.legs if l.kind == "walk"), default=0.0)


class Searcher:
    def __init__(self, g: Graph, cost: CostParams | None = None, max_walk_m: float | None = None):
        self.g = g
        self.cost = cost or CostParams()
        self.max_walk_m = g.params.gap_radius_m if max_walk_m is None else max_walk_m
        c = self.cost
        # "Boardable" edges: pooled bus legs followed by Flex legs. Both carry
        # an effective frequency and are subject to the bottleneck threshold.
        self.n_ride = len(g.ride_u)
        self.b_u = np.concatenate([g.ride_u, g.flex_u])
        self.b_v = np.concatenate([g.ride_v, g.flex_v])
        # Rounded so the bottleneck binary search has a manageable number of levels.
        self.b_freq = np.round(np.concatenate([g.ride_freq, g.flex_freq]), 2)
        self.b_time = np.concatenate([g.ride_time, g.flex_time])
        flex_miles = g.flex_dist / METRES_PER_MILE
        K = g.ride_miles.shape[1]
        fm = np.zeros((len(flex_miles), K))
        fm[:, 1] = flex_miles
        self.b_miles = np.vstack([g.ride_miles, fm])
        headway = g.params.window_s / self.b_freq
        class_pen = np.concatenate([g.ride_miles @ c.class_penalties(g.class_names), flex_miles * c.flex_penalty_s_per_mile])
        self.b_cost = self.b_time + c.wait_factor * headway + c.board_penalty_s + class_pen
        self.walk_cost = g.walk_time * np.where(g.walk_dist > c.long_walk_m, c.long_walk_factor, c.walk_factor)
        self.levels = np.unique(self.b_freq)

    # -- helpers -------------------------------------------------------------

    def _edges(self, threshold: float, max_walk_m: float | None = None):
        g = self.g
        max_walk = self.max_walk_m if max_walk_m is None else max_walk_m
        keep = self.b_freq >= threshold
        wkeep = g.walk_dist <= max_walk
        u = np.concatenate([self.b_u[keep], g.walk_u[wkeep]])
        v = np.concatenate([self.b_v[keep], g.walk_v[wkeep]])
        c = np.concatenate([self.b_cost[keep], self.walk_cost[wkeep]])
        kind = np.concatenate([np.flatnonzero(keep), -1 - np.flatnonzero(wkeep)])
        # csr_matrix sums duplicate entries; keep only the cheapest (u, v).
        key = u * g.n + v
        order = np.lexsort((c, key))
        key, first = np.unique(key[order], return_index=True)
        sel = order[first]
        return u[sel], v[sel], np.maximum(c[sel], 1.0), kind[sel], key

    def _matrix(self, u, v, c):
        return csr_matrix((c, (u, v)), shape=(self.g.n, self.g.n))

    def reachable(self, threshold: float, max_walk_m: float | None = None) -> bool:
        u, v, c, _, _ = self._edges(threshold, max_walk_m)
        order = breadth_first_order(self._matrix(u, v, c), self.g.origin, directed=True, return_predecessors=False)
        return bool(np.isin(self.g.destination, order))

    # -- phase 0: shortest longest walk --------------------------------------

    def required_walk(self) -> float | None:
        """Smallest walk-length limit that connects origin and destination at
        any frequency (None if even the longest modelled walk does not)."""
        g = self.g
        lo_level = float(self.levels[0])
        if self.reachable(lo_level):
            return self.max_walk_m
        cand = np.unique(g.walk_dist[g.walk_dist > self.max_walk_m])
        if not len(cand) or not self.reachable(lo_level, float(cand[-1])):
            return None
        lo, hi = 0, len(cand) - 1  # cand[hi] reachable
        while lo < hi:
            mid = (lo + hi) // 2
            if self.reachable(lo_level, float(cand[mid])):
                hi = mid
            else:
                lo = mid + 1
        return float(cand[hi])

    # -- phase 1: widest path ------------------------------------------------

    def max_bottleneck(self) -> float:
        lv = self.levels
        if not self.reachable(float(lv[0])):
            return 0.0
        lo, hi = 0, len(lv) - 1  # lv[lo] reachable
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.reachable(float(lv[mid])):
                lo = mid
            else:
                hi = mid - 1
        return float(lv[lo])

    # -- phase 2: min generalised cost under a threshold ---------------------

    def best_route(self, threshold: float) -> Route | None:
        g = self.g
        u, v, c, kind, key = self._edges(threshold)
        dist, pred = dijkstra(self._matrix(u, v, c), directed=True, indices=g.origin, return_predecessors=True)
        if not np.isfinite(dist[g.destination]):
            return None
        path = [g.destination]
        while path[-1] != g.origin:
            path.append(int(pred[path[-1]]))
        path.reverse()
        legs = []
        for a, b in zip(path[:-1], path[1:]):
            e = kind[np.searchsorted(key, a * g.n + b)]
            if e >= 0:
                f = float(self.b_freq[e])
                flex = e >= self.n_ride
                legs.append(Leg("flex" if flex else "ride", a, b, float(self.b_time[e]), f, g.params.window_s / f,
                                int(g.flex_zone[e - self.n_ride]) if flex else -1, tuple(float(x) for x in self.b_miles[e])))
            else:
                w = -1 - e
                legs.append(Leg("walk", a, b, float(g.walk_time[w]), dist_m=float(g.walk_dist[w])))
        legs = _merge_walks(legs)
        rides = [l for l in legs if l.kind != "walk"]
        return Route(
            threshold=threshold,
            nodes=path,
            legs=legs,
            cost_s=float(dist[g.destination]),
            bottleneck_freq=min((l.freq for l in rides), default=0.0),
            in_vehicle_s=sum(l.time_s for l in rides),
            walk_s=sum(l.time_s for l in legs if l.kind == "walk"),
            expected_wait_s=sum(0.5 * l.headway_s for l in rides),
            boardings=len(rides),
            window_s=g.params.window_s,
        )

    def frontier(self, best: float, min_gain: float = 0.05) -> list[Route]:
        """Best route at the optimal bottleneck and at each looser rung of the
        headway ladder. A looser rung is listed only if it cuts the
        generalised cost by at least `min_gain` (fraction) versus the last one
        kept."""
        w = self.g.params.window_s
        thresholds = {best, float(self.levels[0])}
        for h in HEADWAY_LADDER_MIN:
            t = round(w / (h * 60), 2)
            if t < best:
                thresholds.add(t)
        out: list[Route] = []
        for t in sorted(thresholds, reverse=True):
            r = self.best_route(t)
            if r is None:
                continue
            if out and r.cost_s > out[-1].cost_s * (1 - min_gain):
                continue
            out.append(r)
        return out


def _merge_walks(legs: list[Leg]) -> list[Leg]:
    out: list[Leg] = []
    for l in legs:
        if out and l.kind == "walk" and out[-1].kind == "walk":
            out[-1] = Leg("walk", out[-1].u, l.v, out[-1].time_s + l.time_s, dist_m=out[-1].dist_m + l.dist_m)
        else:
            out.append(l)
    return out
