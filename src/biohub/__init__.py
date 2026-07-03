"""Biohub cell-tracking-during-development competition package.

Modules:
  io       - zarr image + .geff ground-truth I/O, physical SCALE, TrackGraph
  metric   - local re-implementation of the edge + division Jaccard metric
  detect   - 3D cell-centre detectors (DoG blob, peak, trained U-Net)
  link     - temporal linking (Hungarian, motion-aware), gap closing, pruning
  unet     - 3D U-Net model + Gaussian heatmap target builder
  pipeline - end-to-end Config-driven detect+link, submission writer
"""
from .io import (SCALE, ImageVolume, open_image, TrackGraph, read_geff,
                 list_datasets, embryo_id)

__all__ = [
    "SCALE", "ImageVolume", "open_image", "TrackGraph", "read_geff",
    "list_datasets", "embryo_id",
]

__version__ = "0.1.0"
