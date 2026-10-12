"""Readers for the formats used by this repository.

fbin : [uint32 npts][uint32 dims][npts*dims float32]
ibin : [uint32 npts][uint32 dims][npts*dims uint32]
index: written by VamanaIndex::save()
       [uint32 npts][uint32 dim][uint32 start_node][uint32 n_entry][n_entry uint32]
       then for each node: [uint32 degree][degree uint32 neighbour ids]
"""
import numpy as np


def read_fbin(path, mmap=False):
    npts, dims = np.fromfile(path, dtype=np.uint32, count=2)
    if mmap:
        return np.memmap(path, dtype=np.float32, mode="r", offset=8, shape=(int(npts), int(dims)))
    return np.fromfile(path, dtype=np.float32, offset=8).reshape(int(npts), int(dims))


def read_ibin(path):
    npts, dims = np.fromfile(path, dtype=np.uint32, count=2)
    return np.fromfile(path, dtype=np.uint32, offset=8).reshape(int(npts), int(dims))


def write_ibin(path, arr):
    arr = np.ascontiguousarray(arr, dtype=np.uint32)
    with open(path, "wb") as f:
        np.array(arr.shape, dtype=np.uint32).tofile(f)
        arr.tofile(f)


def read_vamana_index(path):
    """Return (adjacency, start_node, entry_points).

    adjacency is an (npts, max_degree) int32 array padded with -1, which is
    the layout a GPU kernel wants (fixed stride, one gather per hop).
    """
    raw = np.fromfile(path, dtype=np.uint32)
    npts, _dim, start = int(raw[0]), int(raw[1]), int(raw[2])
    nep = int(raw[3])
    entry = raw[4:4 + nep].astype(np.int64)
    pos = 4 + nep
    # First pass: degrees (vectorised walk is awkward with variable lengths,
    # so walk once to collect offsets; 1M nodes takes about a second).
    degs = np.empty(npts, dtype=np.int64)
    offs = np.empty(npts, dtype=np.int64)
    p = pos
    for i in range(npts):
        d = int(raw[p])
        degs[i] = d
        offs[i] = p + 1
        p += 1 + d
    if p != len(raw):
        raise ValueError(f"index file has {len(raw) - p} trailing words; wrong format?")
    max_deg = int(degs.max())
    adj = np.full((npts, max_deg), -1, dtype=np.int32)
    # Fill row by row in blocks of equal degree to keep it vectorised.
    for d in np.unique(degs):
        if d == 0:
            continue
        rows = np.nonzero(degs == d)[0]
        idx = offs[rows][:, None] + np.arange(d)[None, :]
        adj[rows, :d] = raw[idx].astype(np.int32)
    return adj, start, entry
