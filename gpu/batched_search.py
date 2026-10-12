"""Batched Vamana beam search on a GPU (PyTorch).

The C++ search answers one query per thread and spends most of its time
waiting on random memory reads. A GPU hides that latency by running many
queries at once: every step below advances a whole batch of queries by one
hop, so the gathers and distance computations become large, regular tensor
operations.

Per step, for every query in the batch:
  1. pick the `width` closest candidates not yet expanded,
  2. gather their neighbour lists from a padded (N, R) adjacency tensor,
  3. drop neighbours already visited (per-query byte map),
  4. compute exact L2 distances for the new ones in one batched op,
  5. merge them into the sorted candidate list of size L (top-k).
A query is finished when every candidate in its list has been expanded.

width=1 is the same expansion order as the C++ greedy search; larger widths
take fewer, larger steps (better GPU utilisation, a few more distance
computations), as in GPU graph indexes such as CAGRA.

Runs on CPU too (slowly), which is how it was tested.

Example:
  python gpu/batched_search.py --index tmp/idx_impr_r64.bin \\
      --data tmp/sift_base.fbin --queries tmp/sift_query.fbin --gt tmp/sift_gt.ibin \\
      --L 10,20,30,50,75,100 --batch 1024 --width 4
"""
import argparse
import time

import numpy as np
import torch

from ann_io import read_fbin, read_ibin, read_vamana_index

INF = float("inf")


@torch.no_grad()
def search_batch(q, X, x_norm, adj, start, L, K, width, visited):
    """Search one batch. q: (B, D) on device. Returns ids (B, K), dist cmps (B,)."""
    B = q.shape[0]
    dev = q.device
    R = adj.shape[1]
    rows = torch.arange(B, device=dev)
    q_norm = (q * q).sum(1)

    cand_id = torch.full((B, L), -1, dtype=torch.long, device=dev)
    cand_d = torch.full((B, L), INF, device=dev)
    expanded = torch.zeros((B, L), dtype=torch.bool, device=dev)

    s = torch.full((B,), start, dtype=torch.long, device=dev)
    cand_id[:, 0] = s
    cand_d[:, 0] = ((X[s] - q) ** 2).sum(1)
    visited.zero_()
    visited[rows, s] = 1
    cmps = torch.ones(B, dtype=torch.long, device=dev)

    while True:
        # 1. the `width` closest unexpanded candidates per query
        open_d = torch.where(expanded | (cand_id < 0), INF, cand_d)
        sel_d, sel_pos = open_d.topk(width, dim=1, largest=False)
        sel_ok = torch.isfinite(sel_d)                      # (B, W)
        if not bool(sel_ok.any()):
            break
        expanded.scatter_(1, sel_pos, sel_ok | expanded.gather(1, sel_pos))
        nodes = cand_id.gather(1, sel_pos).clamp(min=0)     # (B, W)

        # 2. neighbour lists, (B, W*R)
        nbr = adj[nodes].reshape(B, -1)
        ok = (nbr >= 0) & sel_ok.repeat_interleave(R, dim=1)
        nbr_c = nbr.clamp(min=0)

        # 3. unvisited only; also drop duplicates inside this step
        ok &= visited.gather(1, nbr_c) == 0
        nbr_c = torch.where(ok, nbr_c, torch.zeros_like(nbr_c))
        sorted_n, order = torch.sort(torch.where(ok, nbr_c, torch.full_like(nbr_c, -1)), dim=1)
        dup = torch.zeros_like(ok)
        dup_sorted = torch.zeros_like(ok)
        dup_sorted[:, 1:] = (sorted_n[:, 1:] == sorted_n[:, :-1]) & (sorted_n[:, 1:] >= 0)
        dup.scatter_(1, order, dup_sorted)
        ok &= ~dup
        visited.scatter_add_(1, nbr_c, ok.to(visited.dtype))
        cmps += ok.sum(1)

        # 4. distances ||x||^2 - 2 q.x + ||q||^2 for the new neighbours
        xv = X[nbr_c]                                        # (B, W*R, D)
        d = x_norm[nbr_c] - 2.0 * torch.bmm(xv, q.unsqueeze(2)).squeeze(2) + q_norm[:, None]
        d = torch.where(ok, d.clamp(min=0), torch.full_like(d, INF))

        # 5. merge and keep the best L
        all_d = torch.cat([cand_d, d], 1)
        all_id = torch.cat([cand_id, torch.where(ok, nbr_c, torch.full_like(nbr_c, -1))], 1)
        all_ex = torch.cat([expanded, torch.zeros_like(ok)], 1)
        cand_d, top = all_d.topk(L, dim=1, largest=False)
        cand_id = all_id.gather(1, top)
        expanded = all_ex.gather(1, top)

    return cand_id[:, :K], cmps


def recall_at_k(found, gt, K):
    hits = 0
    for f, g in zip(found, gt[:, :K]):
        hits += len(set(f.tolist()) & set(g.tolist()))
    return hits / (len(found) * K)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--L", default="10,20,30,50,75,100")
    ap.add_argument("--batch", type=int, default=1024, help="queries per batch (visited map uses batch*N bytes)")
    ap.add_argument("--width", type=int, default=4, help="candidates expanded per query per step")
    ap.add_argument("--nq", type=int, default=0, help="use only the first nq queries (0 = all)")
    ap.add_argument("--half", action="store_true", help="store vectors in fp16 (halves memory traffic; small recall cost)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    dev = torch.device(a.device)
    t0 = time.time()
    adj_np, start, _ = read_vamana_index(a.index)
    X_np = read_fbin(a.data)
    Q_np = read_fbin(a.queries)
    gt = read_ibin(a.gt)
    if a.nq:
        Q_np, gt = Q_np[: a.nq], gt[: a.nq]
    dt = torch.float16 if a.half else torch.float32
    X = torch.from_numpy(X_np).to(dev, dt)
    x_norm = (torch.from_numpy(X_np).to(dev).float() ** 2).sum(1).to(dt)
    adj = torch.from_numpy(adj_np.astype(np.int64)).to(dev)
    Q = torch.from_numpy(Q_np).to(dev, dt)
    N = X.shape[0]
    print(f"loaded {N} points, degree<= {adj.shape[1]}, {len(Q)} queries on {dev} in {time.time() - t0:.1f}s")
    visited = torch.zeros((min(a.batch, len(Q)), N), dtype=torch.uint8, device=dev)

    print(f"{'L':>5} {'Recall@'+str(a.K):>10} {'dist cmps':>10} {'QPS':>10} {'batch ms':>10}")
    for L in [int(x) for x in a.L.split(",")]:
        L = max(L, a.K)
        out, cm = [], []
        # warm-up on the first batch so CUDA kernels are compiled/cached
        search_batch(Q[: min(a.batch, len(Q))], X, x_norm, adj, start, L, a.K, a.width, visited)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t = time.time()
        nb = 0
        for i in range(0, len(Q), a.batch):
            qb = Q[i:i + a.batch]
            ids, c = search_batch(qb, X, x_norm, adj, start, L, a.K, a.width, visited[: len(qb)])
            out.append(ids.cpu())
            cm.append(c.cpu())
            nb += 1
        if dev.type == "cuda":
            torch.cuda.synchronize()
        el = time.time() - t
        ids = torch.cat(out).numpy()
        rec = recall_at_k(ids, gt, a.K)
        print(f"{L:>5} {rec:>10.4f} {torch.cat(cm).float().mean().item():>10.1f} "
              f"{len(Q) / el:>10.0f} {1000 * el / nb:>10.1f}")


if __name__ == "__main__":
    main()
