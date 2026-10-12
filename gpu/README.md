# GPU tools

Two PyTorch scripts that put a GPU to work on this index. They read the same files as the C++ tools (`.fbin`, `.ibin`, and the index written by `build_index`). Both also run on CPU (slowly), which is how they were tested.

| Script | What it does |
|---|---|
| `batched_search.py` | Batched beam search over a built Vamana graph. Each step advances a whole batch of queries by one hop: gather neighbour lists from a padded `(N, R)` adjacency tensor, filter visited nodes, compute L2 distances with one batched matmul, merge with top-k. `--width` expands several candidates per query per step (CAGRA-style) for better GPU utilisation. |
| `bruteforce_knn.py` | Exact k-NN by chunked matmul with a running top-k. Generates ground truth for new datasets or subsets, and gives a brute-force throughput baseline (recall 1.0) to compare the graph against. |
| `ann_io.py` | Readers for `.fbin`/`.ibin` and the Vamana index format. |

## Setup

```bash
pip install torch numpy        # a CUDA build of PyTorch
python -c "import torch; print(torch.cuda.is_available())"   # should print True
```

## Run (from the repository root, after building an index)

```bash
# graph search on the GPU: recall, distance computations, throughput
python gpu/batched_search.py --index tmp/idx_impr_r64.bin --data tmp/sift_base.fbin \
    --queries tmp/sift_query.fbin --gt tmp/sift_gt.ibin \
    --L 10,20,30,50,75,100 --batch 1024 --width 4

# exact brute force on the GPU (checks itself against the official ground truth)
python gpu/bruteforce_knn.py --data tmp/sift_base.fbin --queries tmp/sift_query.fbin \
    --K 10 --check tmp/sift_gt.ibin
```

Knobs:
* `--batch`: queries per batch. The visited map costs `batch x N` bytes (1 GB at 1,024 x 1M), so lower it on small GPUs.
* `--width`: candidates expanded per step. `1` reproduces the C++ expansion order; `2`-`8` use the GPU better at the cost of more distance computations.
* `--half`: fp16 vectors (half the memory traffic).

## Correctness check

On SIFT1M with the R=64 index, run on CPU over 200 queries:

| L | width | Recall@10 | Distance comps | C++ `search_index` recall at same L |
|---|---|---|---|---|
| 10 | 1 | 0.9020 | 998 | 0.8865 |
| 50 | 1 | 0.9930 | 2,722 | 0.9916 |
| 10 | 4 | 0.9245 | 1,668 | |
| 50 | 4 | 0.9930 | 3,255 | |

Recall tracks the C++ search; it is a little higher at small L because the C++ search also applies an early-termination heuristic. `bruteforce_knn.py` reproduces the official SIFT1M ground truth exactly.

## Throughput

SIFT1M, R=64 index, all 10,000 queries, `--batch 1024 --width 4`.

| L | Recall@10 | Dist. comps | GPU QPS (laptop NVIDIA GPU) | Same script on CPU, QPS | C++ `search_index`, 1 core, vectorised kernel, QPS (recall) |
|---|---|---|---|---|---|
| 10 | 0.9253 | 1,688 | 40,628 | 2,463 | 9,643 (0.8865) |
| 20 | 0.9695 | 2,093 | 33,125 | 2,099 | 6,640 (0.9567) |
| 30 | 0.9840 | 2,496 | 19,108 | 1,818 | 5,144 (0.9784) |
| 50 | 0.9939 | 3,271 | 18,700 | 1,468 | 3,634 (0.9916) |
| 75 | 0.9974 | 4,168 | 17,174 | 1,242 | 2,660 (0.9963) |
| 100 | 0.9983 | 4,999 | 12,434 | 969 | 2,114 (0.9978) |

* At about 0.99 recall the GPU delivers roughly 5x a single CPU core running the C++ search. A fully loaded multi-core CPU is still competitive.
* Each hop is a handful of separate PyTorch ops, launched from a Python loop, so small batches are launch-bound. Larger `--batch`, fusing a hop into one kernel (Triton/CUDA), or capturing the loop with CUDA Graphs are the next steps.
* `--width 4` expands four candidates per step: fewer steps, at the cost of about 25-70% more distance computations than the sequential C++ search.
