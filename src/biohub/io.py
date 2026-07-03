"""Data I/O for the Biohub cell tracking competition.

Image volumes: Zarr v3, shape (T, Z, Y, X), uint16, one chunk per timepoint at
`0/c/{t}/0/0/0`, blosc/zstd compressed. Metadata in `0/zarr.json`.

Ground truth: `.geff` graph directories (Zarr v3 based) with nodes/props/{t,z,y,x}
and edges/ids (source_id, target_id).
"""
from __future__ import annotations

import os
import json
import itertools
from dataclasses import dataclass

import numpy as np


# Physical voxel scale (z, y, x) in micrometres per voxel.
SCALE = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)


@dataclass
class ImageVolume:
    path: str
    shape: tuple  # (T, Z, Y, X)
    dtype: np.dtype
    chunk: tuple

    @property
    def n_t(self) -> int:
        return int(self.shape[0])

    def frame(self, t: int) -> np.ndarray:
        """Return the (Z, Y, X) volume for timepoint t."""
        return _read_chunk(self.path, t, self.shape, self.dtype)


def open_image(zarr_path: str) -> ImageVolume:
    with open(os.path.join(zarr_path, "0", "zarr.json")) as f:
        meta = json.load(f)
    shape = tuple(int(s) for s in meta["shape"])
    dtype = np.dtype(meta["data_type"])
    chunk = None
    cg = meta.get("chunk_grid", {})
    conf = cg.get("configuration", {})
    if "chunk_shape" in conf:
        chunk = tuple(int(s) for s in conf["chunk_shape"])
    return ImageVolume(path=zarr_path, shape=shape, dtype=dtype, chunk=chunk)


_BLOSC2 = None


def _blosc2():
    global _BLOSC2
    if _BLOSC2 is None:
        import blosc2
        _BLOSC2 = blosc2
    return _BLOSC2


def _read_chunk(zarr_path: str, t: int, shape: tuple, dtype: np.dtype) -> np.ndarray:
    """Read and decode one timepoint chunk -> (Z, Y, X)."""
    frame_shape = shape[1:]
    chunk_path = os.path.join(zarr_path, "0", "c", str(t), "0", "0", "0")
    with open(chunk_path, "rb") as f:
        raw = f.read()
    try:
        dec = _blosc2().decompress(raw)
        arr = np.frombuffer(dec, dtype=dtype)
        if arr.size == int(np.prod(frame_shape)):
            return arr.reshape(frame_shape).copy()
    except Exception:
        pass
    import zarr
    z = zarr.open(os.path.join(zarr_path, "0"), mode="r")
    return np.asarray(z[t])


@dataclass
class TrackGraph:
    """A tracking graph: node coords + directed edges."""
    node_t: np.ndarray
    node_z: np.ndarray
    node_y: np.ndarray
    node_x: np.ndarray
    node_ids: np.ndarray
    edges: np.ndarray
    meta: dict

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def coords_by_id(self) -> dict:
        out = {}
        for i, nid in enumerate(self.node_ids):
            out[int(nid)] = (int(self.node_t[i]), float(self.node_z[i]),
                             float(self.node_y[i]), float(self.node_x[i]))
        return out


def _read_v3_array(arr_dir: str) -> np.ndarray:
    """Read one Zarr v3 array directory (single- or multi-chunk) without needing
    zarr>=3. Supports the ``[bytes, zstd]`` codec pipeline the .geff files use, so
    v3 ground truth is readable on Python 3.10 / zarr 2.x."""
    with open(os.path.join(arr_dir, "zarr.json")) as f:
        meta = json.load(f)
    shape = tuple(int(s) for s in meta["shape"])
    dtype = np.dtype(meta["data_type"])
    conf = meta.get("chunk_grid", {}).get("configuration", {})
    chunk = tuple(int(s) for s in conf.get("chunk_shape", shape))
    sep = (meta.get("chunk_key_encoding", {})
           .get("configuration", {}).get("separator", "/"))
    codecs = [c.get("name") for c in meta.get("codecs", [])]
    fill = meta.get("fill_value", 0)

    def _decode(raw):
        if "zstd" in codecs:
            import numcodecs
            raw = numcodecs.Zstd().decode(raw)
        elif "blosc" in codecs:
            raw = _blosc2().decompress(raw)
        return np.frombuffer(raw, dtype=dtype)

    ndim = len(shape)
    if ndim == 0:
        cpath = os.path.join(arr_dir, "c", "0")
        with open(cpath, "rb") as f:
            return _decode(f.read()).reshape(())
    out = np.full(shape, fill, dtype=dtype)
    n_chunks = [-(-shape[i] // chunk[i]) for i in range(ndim)]
    for cidx in itertools.product(*[range(n) for n in n_chunks]):
        key = sep.join(str(i) for i in cidx)
        cpath = os.path.join(arr_dir, "c", *key.split("/"))
        if not os.path.exists(cpath):
            continue
        with open(cpath, "rb") as f:
            arr = _decode(f.read())
        sl = tuple(slice(cidx[i] * chunk[i],
                         min(cidx[i] * chunk[i] + chunk[i], shape[i]))
                   for i in range(ndim))
        sub = tuple(s.stop - s.start for s in sl)
        arr = arr[: int(np.prod(sub))].reshape(sub)
        out[sl] = arr
    return out


def _read_geff_direct(geff_path: str):
    """Fallback .geff reader that decodes the v3 arrays by hand."""
    def _arr(rel):
        return _read_v3_array(os.path.join(geff_path, *rel.split("/")))

    node_ids = _arr("nodes/ids").astype(np.int64)
    t = _arr("nodes/props/t/values").astype(np.int64)
    z = _arr("nodes/props/z/values").astype(np.float64)
    y = _arr("nodes/props/y/values").astype(np.float64)
    x = _arr("nodes/props/x/values").astype(np.float64)
    edges = _arr("edges/ids").astype(np.int64)
    if edges.ndim == 1:
        edges = edges.reshape(-1, 2)
    return node_ids, t, z, y, x, edges


def read_geff(geff_path: str) -> TrackGraph:
    """Read a .geff ground-truth graph. Tries zarr (v3), falls back to a direct
    v3 decoder so it also works on Python 3.10 / zarr 2.x."""
    try:
        import zarr
        g = zarr.open(geff_path, mode="r")

        def _arr(path):
            return np.asarray(g[path][:])

        node_ids = _arr("nodes/ids").astype(np.int64)
        t = _arr("nodes/props/t/values").astype(np.int64)
        z = _arr("nodes/props/z/values").astype(np.float64)
        y = _arr("nodes/props/y/values").astype(np.float64)
        x = _arr("nodes/props/x/values").astype(np.float64)
        edges = _arr("edges/ids").astype(np.int64)
        if edges.ndim == 1:
            edges = edges.reshape(-1, 2)
        if node_ids.size == 0:
            raise ValueError("empty via zarr; use direct reader")
    except Exception:
        node_ids, t, z, y, x, edges = _read_geff_direct(geff_path)

    meta = {}
    try:
        with open(os.path.join(geff_path, "zarr.json")) as f:
            zj = json.load(f)
        geff_meta = zj.get("attributes", {}).get("geff", {})
        extra = geff_meta.get("extra", {}) or {}
        meta = dict(geff_meta)
        if "estimated_number_of_nodes" in extra:
            meta["estimated_number_of_nodes"] = extra["estimated_number_of_nodes"]
    except Exception:
        pass

    return TrackGraph(node_t=t, node_z=z, node_y=y, node_x=x,
                      node_ids=node_ids, edges=edges, meta=meta)


def list_datasets(root: str, kind: str = "train") -> list:
    """List dataset base-names (without .zarr) under root/<kind>."""
    d = os.path.join(root, kind)
    if not os.path.isdir(d):
        d = root
    names = sorted(n[:-5] for n in os.listdir(d) if n.endswith(".zarr"))
    return names


def embryo_id(dataset_name: str) -> str:
    """Folder names are {embryo_id}_{field_of_view}; embryo is the first segment."""
    return dataset_name.split("_")[0]
