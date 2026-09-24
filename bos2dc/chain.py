"""Evaluate a user-specified chain of routes ("119 -> 1 -> 24 -> ...").

The chain is solved as a shortest path on a *layered* graph: layer k holds
the ride edges of the k-th route (all of its patterns, pooled the same way
as the main graph), and a transfer moves from layer k to layer k+1 at the
same stop or by walking to a stop within `transfer_walk_m`. The cost is the
same generalised cost the main search uses, so boarding and alighting stops
are picked the way the router would pick them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from .geo import project
from .graph import CLASS_ABBREV, Graph, _aggregate, _pattern_pairs, _walk_seconds, pattern_cum_miles
from .search import CostParams, Leg, Route


@dataclass
class ChainStep:
    agency: str | None  # regex; None = any agency
    route: str

    @classmethod
    def parse(cls, token: str, default_agency: str | None) -> "ChainStep":
        if ":" in token:
            agency, route = token.rsplit(":", 1)
            return cls(agency, route)
        return cls(default_agency, token)

    def matches(self, agency: str, route_name: str) -> bool:
        if not re.match(rf"^{re.escape(self.route)}(\s|$)", route_name):
            return False
        return self.agency is None or re.search(self.agency, agency, re.IGNORECASE) is not None


def evaluate_chain(g: Graph, steps: list[ChainStep], origin_xy, dest_xy, cost: CostParams | None = None,
                   transfer_walk_m: float = 800.0, access_m: float = 800.0) -> tuple[Route | None, list[str]]:
    cost = cost or CostParams()
    p = g.params
    n = g.n
    xy = project(g.nodes["lat"], g.nodes["lon"])
    K = len(steps)
    problems = []

    layer_edges = []
    layer_stops = []
    for k, step in enumerate(steps):
        chunks = []
        stops = set()
        for (fid, idx), nodes in g.pattern_nodes.items():
            pat = g.feeds[fid].patterns[idx]
            if not step.matches(pat.agency, pat.route_name):
                continue
            res = _pattern_pairs(nodes, pat.board, pat.alight, pat.dep.astype(np.int64), pat.arr.astype(np.int64), xy, p,
                                 pattern_cum_miles(g, fid, idx, nodes, xy))
            if res is not None:
                chunks.append(res)
                stops |= set(nodes.tolist())
        if not chunks:
            problems.append(f"no service found for route {step.route}" + (f" ({step.agency})" if step.agency else ""))
            layer_edges.append(None)
            layer_stops.append(np.zeros(0, np.int64))
            continue
        u, v, f, t, m = _aggregate(*(np.concatenate([c[i] for c in chunks]) for i in range(5)), n)
        layer_edges.append((u, v, f, t, m))
        layer_stops.append(np.array(sorted(stops), dtype=np.int64))
    if problems:
        return None, problems

    O, D = K * n, K * n + 1
    size = K * n + 2
    eu, ev, ec, ekind = [], [], [], []  # kind: (layer, edge index) for rides, -1 walk
    info = {}
    pen = cost.class_penalties(g.class_names)
    for k, (u, v, f, t, m) in enumerate(layer_edges):
        headway = p.window_s / np.round(f, 2)
        c = t + cost.wait_factor * headway + cost.board_penalty_s + m @ pen
        eu.append(u + k * n); ev.append(v + k * n); ec.append(c)
        ekind.append(np.column_stack([np.full(len(u), k), np.arange(len(u))]))
    walk_kind = []
    # Transfers between consecutive layers.
    for k in range(K - 1):
        a, b = layer_stops[k], layer_stops[k + 1]
        tree = cKDTree(xy[b])
        lists = tree.query_ball_point(xy[a], transfer_walk_m)
        src = np.repeat(a, [len(l) for l in lists])
        dst = b[np.concatenate([np.array(l, dtype=np.int64) for l in lists])] if len(src) else np.zeros(0, np.int64)
        d = np.linalg.norm(xy[src] - xy[dst], axis=1)
        eu.append(src + k * n); ev.append(dst + (k + 1) * n); ec.append(np.maximum(_walk_seconds(d, p) * cost.walk_factor, 1.0))
        walk_kind.append((src, dst, d))
        ekind.append(np.column_stack([np.full(len(src), -1 - k), np.arange(len(src))]))
    # Access and egress.
    for place_xy, layer, is_origin in ((origin_xy, 0, True), (dest_xy, K - 1, False)):
        s = layer_stops[layer]
        d = np.linalg.norm(xy[s] - place_xy, axis=1)
        near = d <= access_m
        if not near.any():
            return None, [f"no stop of route {steps[layer].route} within {access_m:.0f} m of the {'start' if is_origin else 'end'}"]
        nodes, dist = s[near] + layer * n, d[near]
        if is_origin:
            eu.append(np.full(len(nodes), O)); ev.append(nodes)
        else:
            eu.append(nodes); ev.append(np.full(len(nodes), D))
        ec.append(np.maximum(_walk_seconds(dist, p) * cost.walk_factor, 1.0))
        tag = -100 if is_origin else -200
        ekind.append(np.column_stack([np.full(len(nodes), tag), np.arange(len(nodes))]))
        info[tag] = (s[near], dist)

    u = np.concatenate(eu); v = np.concatenate(ev); c = np.concatenate(ec); kind = np.vstack(ekind)
    key = u * size + v
    order = np.lexsort((c, key))
    key, first = np.unique(key[order], return_index=True)
    sel = order[first]
    u, v, c, kind = u[sel], v[sel], c[sel], kind[sel]
    mat = csr_matrix((c, (u, v)), shape=(size, size))
    dist, pred = dijkstra(mat, directed=True, indices=O, return_predecessors=True)
    if not np.isfinite(dist[D]):
        return None, ["the routes do not connect in this order within the transfer walk limit"]
    path = [D]
    while path[-1] != O:
        path.append(int(pred[path[-1]]))
    path.reverse()

    legs: list[Leg] = []
    for a, b in zip(path[:-1], path[1:]):
        e = np.searchsorted(key, a * size + b)
        tag, idx = kind[e]
        if tag >= 0:
            lu, lv, lf, lt, lm = layer_edges[tag]
            f = float(np.round(lf[idx], 2))
            legs.append(Leg("ride", int(lu[idx]), int(lv[idx]), float(lt[idx]), f, p.window_s / f, miles=tuple(float(x) for x in lm[idx])))
        elif tag in (-100, -200):
            stops, dd = info[tag]
            node = int(stops[idx])
            legs.append(Leg("walk", node, node, float(_walk_seconds(dd[idx], p)), dist_m=float(dd[idx])))
        else:
            src, dst, dd = walk_kind[-1 - tag]
            legs.append(Leg("walk", int(src[idx]), int(dst[idx]), float(_walk_seconds(dd[idx], p)), dist_m=float(dd[idx])))
    rides = [l for l in legs if l.kind == "ride"]
    route = Route(
        threshold=0.0, nodes=[x % n for x in path[1:-1]], legs=legs, cost_s=float(dist[D]),
        bottleneck_freq=min(l.freq for l in rides), in_vehicle_s=sum(l.time_s for l in rides),
        walk_s=sum(l.time_s for l in legs if l.kind == "walk"), expected_wait_s=sum(0.5 * l.headway_s for l in rides),
        boardings=len(rides), window_s=p.window_s, label="chain " + " → ".join(s.route for s in steps),
    )
    return route, []


def format_route(g: Graph, r: Route, steps: list[ChainStep] | None = None) -> list[str]:
    """Leg-by-leg text with miles per locality class. With `steps`, ride legs
    are labelled only with the chain's own routes."""
    from .itinerary import fmt_dur, serving_options

    N = g.nodes
    K = len(g.class_names)
    m = r.miles
    tot = m[:K].sum() or 1.0
    abbrev = [CLASS_ABBREV.get(n, n[:2]) for n in g.class_names]
    lines = [
        "  " + " · ".join(f"{n} {v:.0f} mi ({100 * v / tot:.0f}%)" for n, v in zip(g.class_names, m[:K])) + f" · total {m[:K].sum():.0f} mi",
        f"  {r.boardings} boardings · weakest eff. headway {fmt_dur(r.bottleneck_headway_s)} · in-vehicle {fmt_dur(r.in_vehicle_s)}"
        f" · expected wait {fmt_dur(r.expected_wait_s)} · walking {fmt_dur(r.walk_s)} (longest {r.longest_walk_m:.0f} m)",
    ]
    k = 0
    for leg in r.legs:
        if leg.kind == "walk":
            if leg.dist_m >= 50:
                lines.append(f"     walk {leg.dist_m:.0f} m" + (f" → {N.at[leg.v, 'name']}" if leg.u != leg.v else ""))
            continue
        opts = serving_options(g, leg.u, leg.v, int(np.argmax(leg.miles)))
        if steps is not None:
            step = steps[min(k, len(steps) - 1)]
            opts = [o for o in opts if step.matches(o.agency, o.route_name)] or opts
        k += 1
        names = "/".join(sorted({o.route_name for o in opts}))[:24]
        cm = " ".join(f"{a} {x:4.1f}" for a, x in zip(abbrev, leg.miles))
        lines.append(f"     {names:24} {N.at[leg.u, 'name'][:34]:34} → {N.at[leg.v, 'name'][:34]:34} {sum(leg.miles):5.1f} mi ="
                     f" {cm} · eff. headway {fmt_dur(leg.headway_s)} · {fmt_dur(leg.time_s)}")
    return lines
