"""Route search on the frequency graph.

Objective, in priority order:

1. **Bottleneck frequency** (maximise). A chain of buses is only as usable as
   its least frequent leg: that leg sets how many real departure
   opportunities per day the whole trip has and how long you are stranded if
   anything upstream runs late. This is the *widest path* (max-min) problem;
   it is solved exactly by binary search over the distinct leg frequencies
   with a reachability test at each threshold.

2. **Expected journey time** (minimise) among routes that achieve that
   bottleneck: in-vehicle time + `wait_factor` x headway per boarding + a
   fixed boarding penalty + walking. With wait_factor = 0.5 this is the
   expected door-to-door time for a traveller who shows up without
   consulting timetables, i.e. the frequency-based (not schedule-based)
   travel time used in transit assignment models. It still prefers fewer and
   more frequent legs once the bottleneck is fixed.

A lexicographic (bottleneck, cost) label is not order-preserving under edge
extension, so a single Dijkstra pass cannot optimise both; the two-phase
threshold approach is exact. Running phase 2 for a ladder of thresholds
gives the Pareto frontier of (bottleneck headway, expected time).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, dijkstra

from .graph import Graph

HEADWAY_LADDER_MIN = (10, 15, 20, 30, 40, 45, 60, 75, 90, 120, 150, 180, 240, 300, 360, 480, 720)


@dataclass
class CostParams:
    wait_factor: float = 0.5
    board_penalty_s: float = 300.0
    walk_factor: float = 1.5  # walking minutes weigh more than riding ones


@dataclass
class Leg:
    kind: str  # "ride" | "flex" | "walk"
    u: int
    v: int
    time_s: float
    freq: float = 0.0  # effective departures per window (pooled for rides, equivalent for flex)
    headway_s: float = 0.0
    zone: int = -1  # index into Graph.zones for flex legs


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

    @property
    def bottleneck_headway_s(self) -> float:
        return self.window_s / self.bottleneck_freq if self.bottleneck_freq else float("inf")


class Searcher:
    def __init__(self, g: Graph, cost: CostParams | None = None):
        self.g = g
        self.cost = cost or CostParams()
        # "Boardable" edges: pooled bus legs followed by Flex legs. Both carry
        # a departure count and are subject to the bottleneck threshold.
        self.n_ride = len(g.ride_u)
        self.b_u = np.concatenate([g.ride_u, g.flex_u])
        self.b_v = np.concatenate([g.ride_v, g.flex_v])
        # Rounded so the bottleneck binary search has a manageable number of levels.
        self.b_freq = np.round(np.concatenate([g.ride_freq, g.flex_freq]), 2)
        self.b_time = np.concatenate([g.ride_time, g.flex_time])
        headway = g.params.window_s / self.b_freq
        self.b_cost = self.b_time + self.cost.wait_factor * headway + self.cost.board_penalty_s
        self.walk_cost = g.walk_time * self.cost.walk_factor
        self.levels = np.unique(self.b_freq)

    # -- helpers -------------------------------------------------------------

    def _edges(self, threshold: float):
        g = self.g
        keep = self.b_freq >= threshold
        u = np.concatenate([self.b_u[keep], g.walk_u])
        v = np.concatenate([self.b_v[keep], g.walk_v])
        c = np.concatenate([self.b_cost[keep], self.walk_cost])
        kind = np.concatenate([np.flatnonzero(keep), -1 - np.arange(len(g.walk_u))])
        # csr_matrix sums duplicate entries; keep only the cheapest (u, v).
        key = u * g.n + v
        order = np.lexsort((c, key))
        key, first = np.unique(key[order], return_index=True)
        sel = order[first]
        return u[sel], v[sel], np.maximum(c[sel], 1.0), kind[sel], key

    def _matrix(self, u, v, c):
        return csr_matrix((c, (u, v)), shape=(self.g.n, self.g.n))

    def reachable(self, threshold: float) -> bool:
        u, v, c, _, _ = self._edges(threshold)
        order = breadth_first_order(self._matrix(u, v, c), self.g.origin, directed=True, return_predecessors=False)
        return bool(np.isin(self.g.destination, order))

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

    # -- phase 2: min expected time under a threshold ------------------------

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
                                int(g.flex_zone[e - self.n_ride]) if flex else -1))
            else:
                legs.append(Leg("walk", a, b, float(g.walk_time[-1 - e])))
        legs = _merge_walks(legs)
        rides = [l for l in legs if l.kind != "walk"]
        r = Route(
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
        return r

    def frontier(self, best: float, min_gain: float = 0.05) -> list[Route]:
        """Best route at the optimal bottleneck and at each looser rung of the
        headway ladder. A looser rung is listed only if it cuts the expected
        trip time by at least `min_gain` (fraction) versus the last one kept."""
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
            out[-1] = Leg("walk", out[-1].u, l.v, out[-1].time_s + l.time_s)
        else:
            out.append(l)
    return out
