"""Local re-implementation of the competition metric.

Combined score = edge Jaccard (links across time) + division Jaccard (mitosis).

Edge Jaccard (per sample):
  - Predicted nodes matched to GT nodes per timepoint via optimal bipartite
    assignment on scaled centroid distance (<= 7.0 um).
  - A predicted edge is TP iff both endpoints match GT nodes connected by a GT edge.
  - J_E = TP / (TP + FP + FN), adjusted by a penalty on over-predicting the
    total number of nodes (GT is sparse).
  - Per-sample adjusted edge Jaccards are weight-averaged by (TP + FP + FN).

Division Jaccard:
  - A division is a node with >= 2 outgoing edges.
  - For each GT division, the predicted graph is checked for a connected component
    that covers the pre-split stage and touches both daughter lineages.
  - Micro-averaged across all samples.

The exact closed form of the official aggregation is not published; this module
exposes the components so the combination can be calibrated against the LB.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .io import SCALE


MAX_MATCH_UM = 7.0


def match_per_timepoint(pred_coords: dict, gt_coords: dict,
                        max_dist: float = MAX_MATCH_UM) -> dict:
    """Match predicted node ids to GT node ids per timepoint.

    pred_coords / gt_coords: {node_id: (t, z, y, x)} in voxel units.
    Returns {pred_id: gt_id}.
    """
    # group by t
    pred_by_t: dict = {}
    gt_by_t: dict = {}
    for nid, (t, z, y, x) in pred_coords.items():
        pred_by_t.setdefault(t, []).append((nid, z, y, x))
    for nid, (t, z, y, x) in gt_coords.items():
        gt_by_t.setdefault(t, []).append((nid, z, y, x))

    matches: dict = {}
    for t, plist in pred_by_t.items():
        glist = gt_by_t.get(t)
        if not glist:
            continue
        pid = [p[0] for p in plist]
        gid = [g[0] for g in glist]
        pc = np.array([[p[1], p[2], p[3]] for p in plist], dtype=np.float64) * SCALE
        gc = np.array([[g[1], g[2], g[3]] for g in glist], dtype=np.float64) * SCALE
        d = np.sqrt(((pc[:, None, :] - gc[None, :, :]) ** 2).sum(axis=2))
        # gate: set impossible matches very high
        big = max_dist * 1000.0 + 1.0
        cost = np.where(d <= max_dist, d, big)
        ri, ci = linear_sum_assignment(cost)
        for r, c in zip(ri, ci):
            if d[r, c] <= max_dist:
                matches[pid[r]] = gid[c]
    return matches


@dataclass
class EdgeScore:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    n_pred_nodes: int = 0
    n_est_nodes: float = 0.0
    raw_jaccard: float = 0.0
    adj_jaccard: float = 0.0
    weight: float = 0.0


def edge_jaccard(pred_edges, gt_edges, matches, n_pred_nodes, n_est_nodes,
                 over_pred_penalty: bool = True) -> EdgeScore:
    """Compute adjusted edge Jaccard for one sample.

    pred_edges: iterable of (u, v) predicted-node-id pairs.
    gt_edges: iterable of (a, b) gt-node-id pairs.
    matches: {pred_id: gt_id}.
    n_est_nodes: estimated_number_of_nodes (true cell count estimate).
    """
    gt_edge_set = set((int(a), int(b)) for a, b in gt_edges)

    tp = 0
    fp = 0
    covered = set()  # gt edges covered by a TP
    for u, v in pred_edges:
        mu = matches.get(u)
        mv = matches.get(v)
        if mu is not None and mv is not None:
            ge = (int(mu), int(mv))
            if ge in gt_edge_set:
                tp += 1
                covered.add(ge)
            else:
                # both endpoints are annotated cells but no GT edge -> wrong link
                fp += 1
        # edges touching unmatched (possibly unlabeled real) cells are ignored
    fn = len(gt_edge_set - covered)

    denom = tp + fp + fn
    raw = tp / denom if denom > 0 else 0.0

    adj = raw
    if over_pred_penalty and n_est_nodes and n_pred_nodes > n_est_nodes:
        # penalise predicting many more nodes than estimated true count
        adj = raw * (n_est_nodes / n_pred_nodes)

    return EdgeScore(tp=tp, fp=fp, fn=fn, n_pred_nodes=n_pred_nodes,
                     n_est_nodes=float(n_est_nodes or 0.0),
                     raw_jaccard=raw, adj_jaccard=adj, weight=float(denom))


def _out_adj(edges):
    adj: dict = {}
    for u, v in edges:
        adj.setdefault(int(u), []).append(int(v))
    return adj


@dataclass
class DivScore:
    tp: int = 0
    fp: int = 0
    fn: int = 0


def division_score(pred_edges, gt_edges, matches) -> DivScore:
    """Division detection TP/FP/FN for one sample.

    A division = node with >= 2 outgoing edges. A GT division at node a (with
    daughters) is a TP if the predicted graph has a matched node m^{-1}(a) that
    also splits into nodes matched to a's daughters.
    """
    pred_out = _out_adj(pred_edges)
    gt_out = _out_adj(gt_edges)

    # inverse match: gt_id -> pred_id (first wins)
    inv: dict = {}
    for p, g in matches.items():
        inv.setdefault(int(g), int(p))

    gt_divs = {a: ds for a, ds in gt_out.items() if len(ds) >= 2}
    pred_divs = {u: ds for u, ds in pred_out.items() if len(ds) >= 2}

    tp = 0
    matched_pred = set()
    for a, daughters in gt_divs.items():
        p = inv.get(a)
        if p is None or len(pred_out.get(p, [])) < 2:
            continue
        # daughters predicted-matched
        pred_daughter_gts = set()
        for pv in pred_out.get(p, []):
            g = matches.get(pv)
            if g is not None:
                pred_daughter_gts.add(g)
        # need at least two GT daughters covered
        if len(set(daughters) & pred_daughter_gts) >= 2:
            tp += 1
            matched_pred.add(p)

    fn = len(gt_divs) - tp
    fp = len([u for u in pred_divs if u not in matched_pred])
    return DivScore(tp=tp, fp=fp, fn=fn)


@dataclass
class SampleResult:
    name: str
    edge: EdgeScore
    div: DivScore


@dataclass
class Aggregate:
    edge_jaccard: float = 0.0
    division_jaccard: float = 0.0
    combined: float = 0.0
    per_sample: list = field(default_factory=list)
    extras: dict = field(default_factory=dict)


def aggregate(results: list, div_weight: float = 0.5) -> Aggregate:
    """Aggregate per-sample results.

    Edge: weight-averaged by (TP+FP+FN). Division: micro-averaged.
    Combined: convex combination (calibratable).
    """
    w = np.array([r.edge.weight for r in results], dtype=np.float64)
    aj = np.array([r.edge.adj_jaccard for r in results], dtype=np.float64)
    if w.sum() > 0:
        edge_j = float((aj * w).sum() / w.sum())
    else:
        edge_j = 0.0

    dtp = sum(r.div.tp for r in results)
    dfp = sum(r.div.fp for r in results)
    dfn = sum(r.div.fn for r in results)
    ddenom = dtp + dfp + dfn
    div_j = dtp / ddenom if ddenom > 0 else 0.0

    combined = (1 - div_weight) * edge_j + div_weight * div_j
    return Aggregate(edge_jaccard=edge_j, division_jaccard=div_j,
                     combined=combined, per_sample=results,
                     extras={"div_tp": dtp, "div_fp": dfp, "div_fn": dfn})


def score_sample(name, pred_graph, gt_graph, n_est_nodes=None,
                 over_pred_penalty=True) -> SampleResult:
    """Score one sample. pred_graph/gt_graph are TrackGraph-like with
    coords_by_id() and .edges (id pairs). n_est_nodes from gt meta if None.
    """
    pred_coords = pred_graph.coords_by_id()
    gt_coords = gt_graph.coords_by_id()
    matches = match_per_timepoint(pred_coords, gt_coords)

    if n_est_nodes is None:
        n_est_nodes = gt_graph.meta.get("estimated_number_of_nodes") if gt_graph.meta else None
    if not n_est_nodes:
        n_est_nodes = gt_graph.n_nodes

    es = edge_jaccard(
        [(int(a), int(b)) for a, b in pred_graph.edges],
        [(int(a), int(b)) for a, b in gt_graph.edges],
        matches, pred_graph.n_nodes, n_est_nodes, over_pred_penalty,
    )
    ds = division_score(
        [(int(a), int(b)) for a, b in pred_graph.edges],
        [(int(a), int(b)) for a, b in gt_graph.edges],
        matches,
    )
    return SampleResult(name=name, edge=es, div=ds)
