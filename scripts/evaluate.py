#!/usr/bin/env python3
"""Local cross-validation: run the pipeline on train datasets and score against
the .geff ground truth using the local metric re-implementation.

Usage:
    python scripts/evaluate.py --data data/train --config configs/baseline.json \
        --limit 3 --t-limit 20

--limit    : evaluate at most N datasets (speed).
--t-limit  : only use the first N timepoints per dataset (speed).
--div-weight: weight of the division component in the combined score.

Prints per-sample edge/division Jaccard and the aggregate. Use this to calibrate
config changes offline before spending a Kaggle submission.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from biohub import io, read_geff          # noqa: E402
from biohub.pipeline import Config, run_one  # noqa: E402
from biohub import metric                 # noqa: E402


def find_geff(data_dir: str, name: str) -> str | None:
    for cand in (name + ".geff", name + "_gt.geff", name):
        p = os.path.join(data_dir, cand)
        if os.path.isdir(p) and (cand.endswith(".geff") or os.path.isdir(p)):
            if cand.endswith(".geff"):
                return p
    # search
    for f in os.listdir(data_dir):
        if f.startswith(name) and f.endswith(".geff"):
            return os.path.join(data_dir, f)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train")
    ap.add_argument("--config", default="configs/baseline.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--t-limit", type=int, default=None)
    ap.add_argument("--div-weight", type=float, default=0.5)
    ap.add_argument("--names-file", default=None, help="JSON list of dataset names to eval")
    args = ap.parse_args()

    data_dir = os.path.join(ROOT, args.data) if not os.path.isabs(args.data) else args.data
    with open(os.path.join(ROOT, args.config)) as f:
        cfg = Config(**json.load(f))

    names = sorted(d[:-5] for d in os.listdir(data_dir) if d.endswith(".zarr"))
    if args.names_file:
        import json as _j; want=set(_j.load(open(args.names_file)))
        names=[n for n in names if n in want]
    if args.limit:
        names = names[:args.limit]

    results = []
    t0 = time.time()
    for name in names:
        gt_path = find_geff(data_dir, name)
        if gt_path is None:
            print(f"  skip {name}: no .geff GT found")
            continue
        zp = os.path.join(data_dir, name + ".zarr")
        pred = run_one(zp, cfg, t_limit=args.t_limit)
        gt = read_geff(gt_path)
        res = metric.score_sample(name, pred, gt)
        results.append(res)
        print(f"  {name}: edgeJ={res.edge.adj_jaccard:.4f} "
              f"(tp={res.edge.tp} fp={res.edge.fp} fn={res.edge.fn}) "
              f"divJ tp/fp/fn={res.div.tp}/{res.div.fp}/{res.div.fn} "
              f"nodes={pred.n_nodes} ({time.time()-t0:.1f}s)", flush=True)

    if not results:
        sys.exit("No datasets scored. Check --data path and .geff availability.")
    agg = metric.aggregate(results, div_weight=args.div_weight)
    print("\n==== AGGREGATE ====")
    print(f"edge Jaccard    : {agg.edge_jaccard:.4f}")
    print(f"division Jaccard: {agg.division_jaccard:.4f} "
          f"(tp={agg.extras['div_tp']} fp={agg.extras['div_fp']} fn={agg.extras['div_fn']})")
    print(f"combined (w={args.div_weight}): {agg.combined:.4f}")


if __name__ == "__main__":
    main()
