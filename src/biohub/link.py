"""Temporal linking of per-frame detections into a tracking graph."""
import numpy as np
from scipy.optimize import linear_sum_assignment

from .io import SCALE, TrackGraph


def link_frames(frames: list, max_link_um: float = 10.0,
                allow_divisions: bool = False,
                division_max_um: float = 6.0) -> TrackGraph:
    """Link detections across consecutive frames.

    frames: list over t of (M_t, 3) arrays of (z, y, x) voxel coords.
    Returns a predicted TrackGraph with unique node ids.
    """
    node_ids = []
    node_t = []
    node_z = []
    node_y = []
    node_x = []
    frame_ids = []  # per frame: list of assigned global node ids
    nid = 1
    for t, coords in enumerate(frames):
        ids_t = []
        for (z, y, x) in coords:
            node_ids.append(nid)
            node_t.append(t)
            node_z.append(z)
            node_y.append(y)
            node_x.append(x)
            ids_t.append(nid)
            nid += 1
        frame_ids.append(ids_t)

    edges = []
    for t in range(len(frames) - 1):
        a = frames[t]
        b = frames[t + 1]
        if len(a) == 0 or len(b) == 0:
            continue
        ap = a * SCALE
        bp = b * SCALE
        d = np.sqrt(((ap[:, None, :] - bp[None, :, :]) ** 2).sum(axis=2))
        big = max_link_um * 1000.0 + 1.0
        cost = np.where(d <= max_link_um, d, big)
        ri, ci = linear_sum_assignment(cost)
        matched_b = set()
        matched_a = set()
        for r, c in zip(ri, ci):
            if d[r, c] <= max_link_um:
                edges.append((frame_ids[t][r], frame_ids[t + 1][c]))
                matched_a.add(r)
                matched_b.add(c)

        if allow_divisions:
            # try to add a second daughter for parents, from unmatched b nodes
            unmatched_b = [j for j in range(len(b)) if j not in matched_b]
            for r in list(matched_a):
                # the already-linked daughter
                # find nearest unmatched b within division_max_um of parent
                if not unmatched_b:
                    break
                pr = ap[r]
                dd = np.sqrt(((pr[None, :] - bp[unmatched_b]) ** 2).sum(axis=1))
                j = int(np.argmin(dd))
                if dd[j] <= division_max_um:
                    bj = unmatched_b[j]
                    edges.append((frame_ids[t][r], frame_ids[t + 1][bj]))
                    unmatched_b.remove(bj)

    g = TrackGraph(
        node_t=np.array(node_t, dtype=np.int64),
        node_z=np.array(node_z, dtype=np.float64),
        node_y=np.array(node_y, dtype=np.float64),
        node_x=np.array(node_x, dtype=np.float64),
        node_ids=np.array(node_ids, dtype=np.int64),
        edges=np.array(edges, dtype=np.int64).reshape(-1, 2),
        meta={},
    )
    return g


def close_gaps(frames: list, g: TrackGraph, max_gap: int = 1,
               gap_dist_um: float = 8.0) -> TrackGraph:
    """Insert interpolated nodes to bridge single-frame detection gaps.

    For track-ends at frame t with no successor and track-starts at frame t+2
    with no predecessor, if they are within gap_dist_um*2, insert an interpolated
    node at t+1 (midpoint) and connect end -> interp -> start. Recovers cells
    missed in one frame (all GT edges are dt=1, so this yields 2 matchable edges).
    """
    if g.n_edges == 0:
        return g
    coords = {int(nid): (int(g.node_t[i]), g.node_z[i], g.node_y[i], g.node_x[i])
              for i, nid in enumerate(g.node_ids)}
    has_out = set(int(s) for s, _ in g.edges)
    has_in = set(int(t) for _, t in g.edges)
    # candidates by frame
    ends_by_t = {}   # t -> list of node ids with no outgoing edge
    starts_by_t = {}  # t -> list of node ids with no incoming edge
    for nid, (t, z, y, x) in coords.items():
        if nid not in has_out:
            ends_by_t.setdefault(t, []).append(nid)
        if nid not in has_in:
            starts_by_t.setdefault(t, []).append(nid)

    new_nodes = []  # (t, z, y, x)
    new_edges = []
    next_id = int(g.node_ids.max()) + 1 if g.n_nodes else 1
    for gap in range(1, max_gap + 1):
        for t, ends in ends_by_t.items():
            starts = starts_by_t.get(t + gap + 1)
            if not starts:
                continue
            ec = np.array([[coords[e][1], coords[e][2], coords[e][3]] for e in ends]) * SCALE
            sc = np.array([[coords[s][1], coords[s][2], coords[s][3]] for s in starts]) * SCALE
            d = np.sqrt(((ec[:, None, :] - sc[None, :, :]) ** 2).sum(axis=2))
            thr = gap_dist_um * (gap + 1)
            big = thr * 1000 + 1
            cost = np.where(d <= thr, d, big)
            ri, ci = linear_sum_assignment(cost)
            used_s = set()
            for r, c in zip(ri, ci):
                if d[r, c] > thr or ends[r] in has_out or starts[c] in used_s:
                    continue
                e_id, s_id = ends[r], starts[c]
                te, ze, ye, xe = coords[e_id]
                ts, zs, ys, xs = coords[s_id]
                prev = e_id
                for k in range(1, gap + 1):
                    frac = k / (gap + 1)
                    zi = ze + (zs - ze) * frac
                    yi = ye + (ys - ye) * frac
                    xi = xe + (xs - xe) * frac
                    nid = next_id
                    next_id += 1
                    new_nodes.append((te + k, zi, yi, xi, nid))
                    new_edges.append((prev, nid))
                    prev = nid
                new_edges.append((prev, s_id))
                has_out.add(e_id)
                used_s.add(c)
    if not new_nodes:
        return g
    nt = np.concatenate([g.node_t, np.array([n[0] for n in new_nodes], dtype=np.int64)])
    nz = np.concatenate([g.node_z, np.array([n[1] for n in new_nodes])])
    ny = np.concatenate([g.node_y, np.array([n[2] for n in new_nodes])])
    nx = np.concatenate([g.node_x, np.array([n[3] for n in new_nodes])])
    nid = np.concatenate([g.node_ids, np.array([n[4] for n in new_nodes], dtype=np.int64)])
    edges = np.concatenate([g.edges, np.array(new_edges, dtype=np.int64).reshape(-1, 2)])
    return TrackGraph(node_t=nt, node_z=nz, node_y=ny, node_x=nx, node_ids=nid,
                      edges=edges, meta=g.meta)


def link_motion(frames: list, max_link_um: float = 8.0,
                motion_weight: float = 0.5, max_miss: int = 0) -> TrackGraph:
    """Velocity-aware incremental tracking.

    Each active track predicts its next position from its last velocity; the
    assignment cost is the distance to that prediction. Reduces mismatches for
    moving cells in dense regions vs. plain nearest-neighbour linking.

    motion_weight blends predicted position (1.0) vs last position (0.0).
    max_miss: allow a track to survive this many frames without a detection
    (gap tolerance), predicting forward.
    """
    node_ids = []
    node_t = []
    node_z = []
    node_y = []
    node_x = []
    frame_ids = []
    nid = 1
    for t, coords in enumerate(frames):
        ids_t = []
        for (z, y, x) in coords:
            node_ids.append(nid); node_t.append(t); node_z.append(z)
            node_y.append(y); node_x.append(x); ids_t.append(nid); nid += 1
        frame_ids.append(ids_t)

    edges = []
    # active tracks: dict track -> {last_pos(voxel), vel(voxel), last_node_id, miss}
    active = {}
    tid = 0
    if len(frames) and len(frames[0]):
        for j, (z, y, x) in enumerate(frames[0]):
            active[tid] = {"pos": np.array([z, y, x], float),
                           "vel": np.zeros(3), "node": frame_ids[0][j], "miss": 0}
            tid += 1

    for t in range(1, len(frames)):
        b = frames[t]
        if len(b) == 0:
            for tr in active.values():
                tr["miss"] += 1
            active = {k: v for k, v in active.items() if v["miss"] <= max_miss}
            for tr in active.values():
                tr["pos"] = tr["pos"] + tr["vel"]
            continue
        akeys = list(active.keys())
        if akeys:
            pred = np.array([active[k]["pos"] + motion_weight * active[k]["vel"]
                             for k in akeys])
            predp = pred * SCALE
            bp = b * SCALE
            d = np.sqrt(((predp[:, None, :] - bp[None, :, :]) ** 2).sum(axis=2))
            big = max_link_um * 1000 + 1
            cost = np.where(d <= max_link_um, d, big)
            ri, ci = linear_sum_assignment(cost)
        else:
            ri, ci = [], []
        matched_b = set()
        matched_tr = set()
        for r, c in zip(ri, ci):
            if d[r, c] <= max_link_um:
                k = akeys[r]
                tr = active[k]
                newpos = np.array(b[c], float)
                edges.append((tr["node"], frame_ids[t][c]))
                tr["vel"] = newpos - tr["pos"]
                tr["pos"] = newpos
                tr["node"] = frame_ids[t][c]
                tr["miss"] = 0
                matched_b.add(c); matched_tr.add(k)
        # unmatched existing tracks: age or drop
        drop = []
        for k in akeys:
            if k not in matched_tr:
                active[k]["miss"] += 1
                active[k]["pos"] = active[k]["pos"] + active[k]["vel"]
                if active[k]["miss"] > max_miss:
                    drop.append(k)
        for k in drop:
            del active[k]
        # new tracks for unmatched detections
        for c in range(len(b)):
            if c not in matched_b:
                active[tid] = {"pos": np.array(b[c], float), "vel": np.zeros(3),
                               "node": frame_ids[t][c], "miss": 0}
                tid += 1

    g = TrackGraph(
        node_t=np.array(node_t, dtype=np.int64),
        node_z=np.array(node_z, dtype=np.float64),
        node_y=np.array(node_y, dtype=np.float64),
        node_x=np.array(node_x, dtype=np.float64),
        node_ids=np.array(node_ids, dtype=np.int64),
        edges=np.array(edges, dtype=np.int64).reshape(-1, 2),
        meta={},
    )
    return g


def prune_isolated(g: TrackGraph) -> TrackGraph:
    """Remove nodes not referenced by any edge."""
    if g.n_edges == 0:
        return g
    used = set(int(x) for x in g.edges.reshape(-1))
    keep = np.array([i for i, nid in enumerate(g.node_ids) if int(nid) in used])
    if len(keep) == len(g.node_ids):
        return g
    return TrackGraph(
        node_t=g.node_t[keep], node_z=g.node_z[keep], node_y=g.node_y[keep],
        node_x=g.node_x[keep], node_ids=g.node_ids[keep], edges=g.edges, meta=g.meta,
    )
