"""Exact k-NN on a GPU: ground truth for new datasets, and a brute-force
throughput baseline to compare the graph index against.

Distances use ||x||^2 - 2 q.x (||q||^2 is constant per query, so it does not
change the ranking). Base vectors are processed in chunks so 1M x 128 fits in
a few GB; candidates from each chunk are merged with a running top-k.

Examples:
  # ground truth (K=100) for a dataset, written in ibin format
  python gpu/bruteforce_knn.py --data tmp/sift_base.fbin --queries tmp/sift_query.fbin \\
      --K 100 --out tmp/sift_gt_gpu.ibin
  # check it against the official ground truth, and time it
  python gpu/bruteforce_knn.py --data tmp/sift_base.fbin --queries tmp/sift_query.fbin \\
      --K 10 --check tmp/sift_gt.ibin
"""
import argparse
import time

import numpy as np
import torch

from ann_io import read_fbin, read_ibin, write_ibin


@torch.no_grad()
def knn(X, Q, K, chunk=262144, qbatch=4096):
    dev = X.device
    x_norm = (X.float() ** 2).sum(1)
    out_d, out_i = [], []
    for qs in range(0, len(Q), qbatch):
        q = Q[qs:qs + qbatch].float()
        best_d = torch.full((len(q), K), float("inf"), device=dev)
        best_i = torch.full((len(q), K), -1, dtype=torch.long, device=dev)
        for cs in range(0, len(X), chunk):
            xc = X[cs:cs + chunk].float()
            d = x_norm[cs:cs + chunk][None, :] - 2.0 * q @ xc.T
            kk = min(K, d.shape[1])
            cd, ci = d.topk(kk, dim=1, largest=False)
            md = torch.cat([best_d, cd], 1)
            mi = torch.cat([best_i, ci + cs], 1)
            best_d, top = md.topk(K, dim=1, largest=False)
            best_i = mi.gather(1, top)
        out_d.append(best_d)
        out_i.append(best_i)
    return torch.cat(out_d), torch.cat(out_i)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--K", type=int, default=100)
    ap.add_argument("--out", help="write ground truth here (ibin)")
    ap.add_argument("--check", help="compare against this ibin ground truth")
    ap.add_argument("--nq", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    # Exact fp32 matmul: TF32 would trade exactness for speed.
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device(a.device)
    X = torch.from_numpy(read_fbin(a.data)).to(dev)
    Q = torch.from_numpy(read_fbin(a.queries)).to(dev)
    if a.nq:
        Q = Q[: a.nq]
    knn(X, Q[:16], a.K)  # warm-up
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t = time.time()
    _, ids = knn(X, Q, a.K)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    el = time.time() - t
    ids = ids.cpu().numpy()
    print(f"{len(Q)} queries x {len(X)} points, K={a.K}: {el:.2f}s ({len(Q) / el:.0f} QPS) on {dev}")
    if a.check:
        gt = read_ibin(a.check)[: len(Q)]
        k = min(a.K, gt.shape[1])
        hits = sum(len(set(r[:k]) & set(g[:k])) for r, g in zip(ids, gt))
        print(f"recall@{k} vs {a.check}: {hits / (len(Q) * k):.4f}  (ties can make this a hair below 1.0)")
    if a.out:
        write_ibin(a.out, ids)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
