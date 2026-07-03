#!/usr/bin/env python3
"""Concatenate the ``biohub`` package into ONE self-contained Kaggle inference
kernel (no package imports), the shape Kaggle code competitions expect.

Usage:
    python scripts/build_kernel.py --config configs/baseline.json \
        --out kernels/inference_kernel.py

The generated file inlines io/metric/detect/link/unet/pipeline (relative imports
removed), aliases ``io`` to the current module so intra-file ``io.X`` calls
resolve, and appends an inference driver that writes ``submission.csv``.
"""
from __future__ import annotations
import argparse
import json
import os
import re

MODULES = ["io", "metric", "detect", "link", "unet", "pipeline"]

HEADER = '''from __future__ import annotations
# -*- coding: utf-8 -*-
# AUTO-GENERATED self-contained Kaggle inference kernel. Do not edit by hand.
# Regenerate with: python scripts/build_kernel.py --config <cfg> --out <path>
import os, sys, json, time
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from scipy.ndimage import (gaussian_filter, maximum_filter, grey_erosion,
                           grey_dilation)
from scipy.optimize import linear_sum_assignment
try:
    import blosc2
except Exception:
    blosc2 = None
'''

DRIVER = '''

import sys as _sys
io = _sys.modules[__name__]

CONFIG_OVERRIDE = __CONFIG__


# ============================ inference driver ============================
def find_test_dir():
    env = os.environ.get("TEST_DIR")
    cands = [
        env,
        "/kaggle/input/biohub-cell-tracking-during-development/test",
        "/kaggle/input/competitions/biohub-cell-tracking-during-development/test",
    ]
    for c in cands:
        if c and os.path.isdir(c):
            return c
    base = "/kaggle/input"
    if os.path.isdir(base):
        for root, dirs, files in os.walk(base):
            if os.path.basename(root) == "test" and any(d.endswith(".zarr") for d in dirs):
                return root
    raise FileNotFoundError("test dir not found")


def main():
    test_dir = find_test_dir()
    names = sorted(d[:-5] for d in os.listdir(test_dir) if d.endswith(".zarr"))
    print(f"test dir: {test_dir}; {len(names)} datasets", flush=True)
    cfg = Config(**CONFIG_OVERRIDE)
    unet_model, unet_device = (None, "cuda")
    if cfg.detector in ("unet", "hybrid"):
        unet_model, unet_device = _load_unet(cfg)
    elif cfg.detector == "ens":
        m1, d1 = _load_unet(cfg)
        m2, d2 = _load_unet(cfg, ckpt=cfg.unet_ckpt2)
        unet_model = [(m1, d1, cfg.xy_downsample), (m2, d2, cfg.xy_downsample2)]
        unet_device = d1
    all_rows = []
    t0 = time.time()
    for i, name in enumerate(names):
        zp = os.path.join(test_dir, name + ".zarr")
        g = run_one(zp, cfg, unet_model=unet_model, unet_device=unet_device)
        all_rows.extend(graph_to_rows(name, g))
        print(f"[{i+1}/{len(names)}] {name}: nodes={g.n_nodes} edges={g.n_edges} "
              f"({time.time()-t0:.1f}s)", flush=True)
    out = "submission.csv"
    write_submission(all_rows, out)
    print(f"wrote {out}: {len(all_rows)} rows in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
'''


def strip_module(src):
    out = []
    skip_paren = 0
    for line in src.splitlines():
        s = line.rstrip("\n")
        if skip_paren:
            skip_paren += s.count("(") - s.count(")")
            continue
        if re.match(r"^\s*from \.", s):
            skip_paren = s.count("(") - s.count(")")
            continue
        if re.match(r"^from __future__", s):
            continue
        out.append(s)
    return "\n".join(out)


def build(pkg_dir, config):
    parts = [HEADER]
    for m in MODULES:
        with open(os.path.join(pkg_dir, m + ".py")) as f:
            src = f.read()
        parts.append("\n# ===== biohub." + m + " =====\n" + strip_module(src))
    parts.append(DRIVER.replace("__CONFIG__", repr(config)))
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.json")
    ap.add_argument("--out", default="kernels/inference_kernel.py")
    ap.add_argument("--pkg", default="src/biohub")
    args = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, args.config)) as f:
        config = json.load(f)
    text = build(os.path.join(root, args.pkg), config)
    out_path = os.path.join(root, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(text)
    print("wrote " + out_path + " (" + str(len(text)) + " bytes) from " + args.config)


if __name__ == "__main__":
    main()
