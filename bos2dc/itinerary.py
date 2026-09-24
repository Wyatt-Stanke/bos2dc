"""Timetable check of a frequency-chosen route.

The route is picked on frequencies; this module replays it against the actual
trips to answer "if I leave South Station at time t, when do I reach Union
Station?". Sweeping t across a day yields the route's *connection profile*:
the distinct (latest departure, arrival) pairs, i.e. how many genuinely
different end-to-end journeys the chain offers per day.

Each feed contributes the timetable of one representative service day (the
same weekday in every feed). Journeys that run past midnight assume that
timetable repeats on the following day.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geo import project
from .graph import Graph, pattern_cum_miles
from .search import Route

DAY = 86400


@dataclass
class RideOption:
    feed_id: str
    pattern: int
    route_name: str
    agency: str
    headsign: str
    dep: np.ndarray  # departures from boarding stop, seconds
    arr: np.ndarray  # arrivals at alighting stop (same trips)
    miles: np.ndarray  # miles per locality class between the two stops


@dataclass
class Step:
    kind: str
    start: int
    end: int
    label: str


def serving_options(g: Graph, u: int, v: int, cls: int | None = None) -> list[RideOption]:
    """Trip patterns that serve u -> v (optionally only those whose dominant
    locality class on that segment is `cls`, matching how ride edges are
    pooled)."""
    xy = getattr(g, "_xy", None)
    if xy is None or len(xy) != g.n:
        xy = g._xy = project(g.nodes["lat"], g.nodes["lon"])
    opts = []
    for (feed_id, k), nodes in g.pattern_nodes.items():
        iu = np.flatnonzero(nodes == u)
        if not len(iu):
            continue
        iv = np.flatnonzero(nodes == v)
        if not len(iv):
            continue
        pat = g.feeds[feed_id].patterns[k]
        for i in iu:
            if not pat.board[i]:
                continue
            js = iv[(iv > i) & pat.alight[iv]]
            if len(js):
                j = js[0]
                cum = pattern_cum_miles(g, feed_id, k, nodes, xy)
                miles = cum[j] - cum[i]
                if cls is None or int(np.argmax(miles)) == cls:
                    opts.append(RideOption(feed_id, k, pat.route_name, pat.agency, pat.headsign,
                                           pat.dep[:, i].astype(np.int64), pat.arr[:, j].astype(np.int64), miles))
                break
    return opts


class Simulator:
    def __init__(self, g: Graph, route: Route, transfer_slack_s: int = 120):
        self.g = g
        self.route = route
        self.slack = transfer_slack_s
        self.options = [serving_options(g, l.u, l.v, int(np.argmax(l.miles))) if l.kind == "ride" else None for l in route.legs]
        # A Flex vehicle is assumed to arrive half an equivalent headway after booking.
        self.flex_response = g.params.flex_headway_s // 2

    def _flex(self, leg, ready: int):
        zone = self.g.zones[leg.zone]
        windows = zone.windows(self.g.day)
        day = ready // DAY
        for k in range(day, day + 8):
            for s0, s1, _ in windows:
                pickup = max(ready + self.flex_response, s0 + k * DAY)
                if pickup <= s1 + k * DAY:
                    return pickup + int(round(leg.time_s)), pickup
        return None

    def _ride(self, opts: list[RideOption], ready: int):
        """Earliest arrival: (arrival, departure, option, trip row, day offset)."""
        best = None
        day = ready // DAY
        for o in opts:
            # Service-day times may exceed 24h; try the previous day's late trips too.
            for k in range(day - 1, day + 3):
                off = k * DAY
                ok = o.dep + off >= ready
                if not ok.any():
                    continue
                idx = np.flatnonzero(ok)
                a = o.arr[idx] + off
                m = int(np.argmin(a))
                cand = (int(a[m]), int(o.dep[idx[m]] + off), o, int(idx[m]), off)
                if best is None or cand[0] < best[0] or (cand[0] == best[0] and cand[1] > best[1]):
                    best = cand
                break
        return best

    @staticmethod
    def _stay(opts: list[RideOption], on_board, t: int):
        if on_board is None:
            return None
        fid, pat, row, off = on_board
        for o in opts:
            if o.feed_id == fid and o.pattern == pat and o.dep[row] + off >= t:
                return int(o.arr[row] + off), int(o.dep[row] + off), o, row, off
        return None

    def run(self, t0: int, detail: bool = False):
        t = t0
        steps: list[Step] = []
        rode = False
        on_board = None  # (feed_id, pattern, trip row, day offset) of the bus just ridden
        for leg, opts in zip(self.route.legs, self.options):
            if leg.kind == "walk":
                end = t + int(round(leg.time_s))
                if detail:
                    steps.append(Step("walk", t, end, f"walk to {self.g.nodes.at[leg.v, 'name']}"))
                t = end
                on_board = None
                continue
            if leg.kind == "flex":
                res = self._flex(leg, t)
                if res is None:
                    return None, steps
                arr, dep = res
                on_board = None
                label = f"{self.g.zones[leg.zone].agency} {self.g.zones[leg.zone].name} (on demand) to {self.g.nodes.at[leg.v, 'name']}"
            else:
                best = self._ride(opts, t + (self.slack if rode else 0))
                # Staying on the same bus needs no transfer slack.
                stay = self._stay(opts, on_board, t)
                if stay is not None and (best is None or stay[0] <= best[0]):
                    best = stay
                if best is None:
                    return None, steps
                arr, dep, o, row, off = best
                on_board = (o.feed_id, o.pattern, row, off)
                label = f"{o.agency} {o.route_name}" + (f" toward {o.headsign}" if o.headsign else "")
            if detail:
                steps.append(Step("wait", t, dep, ""))
                steps.append(Step(leg.kind, dep, arr, label))
            t = arr
            rode = True
        return t, steps

    def profile(self, step_s: int = 60):
        """Distinct (latest departure from origin, arrival) connections whose
        latest departure falls within one day. The sweep covers two days so a
        departure late in the day is attributed to the connection it really
        catches (whose latest departure may be the next afternoon)."""
        arrivals = {}
        for t0 in range(0, 2 * DAY, step_s):
            arr, _ = self.run(t0)
            if arr is None:
                continue
            arrivals[arr] = t0  # ascending t0: keeps the latest departure
        conns = sorted((dep, arr) for arr, dep in arrivals.items() if dep < DAY)
        # Drop dominated connections (leave earlier, arrive later).
        out = []
        for dep, arr in reversed(conns):
            if not out or arr < out[-1][1]:
                out.append((dep, arr))
        return list(reversed(out))


def fmt_clock(t: int) -> str:
    d, rem = divmod(int(t), DAY)
    h, rem = divmod(rem, 3600)
    m = rem // 60
    s = f"{h:02d}:{m:02d}"
    return s if d == 0 else f"{s} (+{d}d)"


def fmt_dur(s: float) -> str:
    s = int(round(s))
    h, rem = divmod(s, 3600)
    return f"{h}h{rem // 60:02d}m" if h else f"{rem // 60}m"
