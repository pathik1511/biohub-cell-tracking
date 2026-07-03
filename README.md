# Biohub — Cell Tracking During Development

Toolkit for the Kaggle [Biohub Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)
competition — detect and track zebrafish cells through 3D space and time.

> **Note:** this public repo holds the shared **infrastructure and baseline**
> (data I/O, a local re-implementation of the competition metric, tracking
> primitives, and build/eval tooling). The competitive solution is developed
> privately and is intentionally not part of this repository.

## Layout

```
src/biohub/
  io.py        zarr image + .geff ground-truth I/O, physical SCALE, TrackGraph
  metric.py    local re-implementation of the edge + division Jaccard metric
  link.py      temporal linking (Hungarian + motion-aware), gap closing, pruning
scripts/
  download_data.py  pull competition data via the Kaggle API
  build_kernel.py   package -> single-file Kaggle inference kernel
  evaluate.py       local CV: run a config on train and score vs .geff GT
configs/
  baseline.json     the rule-based baseline configuration
kernels/
  inference_baseline_original.py   self-contained rule-based baseline kernel
```

## Quickstart

```bash
# 1. environment
python -m venv .venv && . .venv/Scripts/activate   # (Windows) or bin/activate
pip install -r requirements.txt

# 2. data (needs a Kaggle API token at ~/.kaggle/kaggle.json; git-ignored)
python scripts/download_data.py

# 3. score a config locally against the .geff ground truth
python scripts/evaluate.py --config configs/baseline.json --limit 3
```

## The metric (`src/biohub/metric.py`)

The competition score is dominated by an **edge Jaccard**: predicted nodes are
matched to ground-truth nodes per timepoint by optimal assignment on physical
centroid distance (≤ 7 µm), and a predicted edge is a true positive only if both
endpoints match GT nodes joined by a GT edge. The score is adjusted by an
over-prediction penalty (`n_est / n_pred`) when more nodes are predicted than the
estimated true cell count. `metric.py` reproduces this locally so configurations
can be compared offline.

## Git & GitHub

See `scripts/init_git.sh` to initialise the repo, push to GitHub, and create a
Projects board. The ignore rules in `.gitignore` keep private assets (weights,
data, and the competitive solution) out of the repository.
