# GraphANN — Vamana Index for Approximate Nearest Neighbor Search

**DA2303 — Algorithms for Data Science, Project Milestone 2**
---

## Overview

This project implements **Vamana**, the graph-based approximate nearest neighbor (ANN) algorithm from the NeurIPS 2019 paper *DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node*. It builds a navigable directed graph over 1M 128-dimensional SIFT vectors and answers queries in under a millisecond at 99%+ recall. It also includes an HNSW implementation for comparison, a thread-safe parallel build, and PyTorch tools for running search on a GPU.

### Key Results

SIFT1M, recall@10, single-threaded search. The table compares the cheapest beam width `L` that reaches each recall target. Full tables and setup are in [results/sift1m.md](results/sift1m.md).

| Recall@10 target | Baseline graph (R=32) | Improved graph (R=64) | Improved graph + vectorised L2 kernel |
|---|---|---|---|
| >= 0.99 | L=150: 528 µs mean / 760 µs P99 | L=50: 349 / 459 (**-34% / -40%**) | L=50: 275 / 363 (**-48% / -52%**) |
| >= 0.995 | L=200: 661 / 973 | L=75: 469 / 631 (-29% / -35%) | L=75: 376 / 503 (-43% / -48%) |

* The build improvements (medoid start, random R-regular initialisation, full visited set into RobustPrune, R=64) pay off at high recall. At ~0.98 recall the larger graph does not help.
* Letting the compiler vectorise the float reduction in the L2 kernel (`-DGRAPHANN_SIMD_REDUCTION=ON`) speeds up every configuration by about 20% (17-26%) with identical results.
* uint8 ADC traversal cuts vector memory 4x (488 to 122 MiB) but is 17-44% *slower* than exact float32 without hand-written SIMD (see Negative Results).
* Batched GPU search (`gpu/batched_search.py`, 1,024 queries per batch) reaches 18,700 QPS at 0.994 recall@10, about 5x one CPU core running the C++ search at similar recall.

---

## Repository Structure

```
graphann/
├── CMakeLists.txt              # Build system (5 executables: Vamana + HNSW)
├── README.md                   # This file
├── RESULTS.txt                 # HNSW vs Vamana comparison benchmarks
├── results/
│   └── sift1m.md               # Full SIFT1M tables (recall, latency, P99, QPS per L)
├── include/
│   ├── distance.h              # 5 distance functions (L2², ADC, PCA, early-abandon)
│   ├── io_utils.h              # fbin/ibin file I/O + aligned memory
│   ├── timer.h                 # Chrono stopwatch
│   ├── vamana_index.h          # VamanaIndex class + GraphStats
│   └── hnsw_index.h            # HNSWIndex class (multi-layer graph)
├── src/
│   ├── distance.cpp            # Distance implementations
│   ├── io_utils.cpp            # File loader implementations
│   ├── vamana_index.cpp        # Core Vamana: greedy_search, robust_prune, build
│   ├── hnsw_index.cpp          # Core HNSW: search_layer, select_neighbors, build
│   ├── build_index.cpp         # CLI: build Vamana index
│   ├── search_index.cpp        # CLI: search Vamana + recall/latency evaluation
│   ├── build_hnsw.cpp          # CLI: build HNSW index
│   ├── search_hnsw.cpp         # CLI: search HNSW + recall/latency evaluation
│   └── hard_query_analysis.cpp # CLI: hard query characterization tool
├── scripts/
│   ├── convert_vecs.py         # fvecs/ivecs → fbin/ibin converter
│   ├── run_sift1m.sh           # End-to-end pipeline script (Linux/macOS)
│   ├── run_ablation.sh         # Build-configuration ablation (8 conditions)
│   ├── benchmark_sift1m.sh     # Builds the three reference indexes and benchmarks every search mode
│   ├── summarize_search_logs.py # Turns search logs into the tables in results/
│   └── analyze_results.py      # Plot generation (Pareto, histograms)
├── gpu/
│   ├── batched_search.py       # Batched Vamana beam search on a GPU (PyTorch)
│   ├── bruteforce_knn.py       # Exact GPU k-NN: ground truth + brute-force baseline
│   └── ann_io.py               # Readers for fbin/ibin and the index format
├── report/
│   ├── main.tex                # LaTeX project report
│   └── *.png                   # Experiment plots and degree histograms
├── run_benchmarks.bat          # HNSW + Vamana benchmark suite (Windows)
├── run_benchmarks.ps1          # PowerShell benchmark suite
├── run_full_benchmark.bat      # Quick comparison benchmark (Windows)
└── run_full_benchmark.ps1      # PowerShell quick benchmark
```

---

## Build & Run

### Prerequisites
- C++17 compiler (GCC 7+, Clang 8+, or MSVC 2019+)
- CMake 3.14+
- OpenMP (usually bundled with the compiler)
- Python 3 + NumPy (for data conversion and plotting)

### Quick Start

```bash
# 1. Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 2. Download and convert SIFT1M (run from project root)
./scripts/run_sift1m.sh

# 3. Or manually:
#    Build index
./build/build_index \
  --data tmp/sift_base.fbin \
  --output tmp/sift_index.bin \
  --R 64 --L 100 --alpha 1.2 --gamma 1.5 \
  --single_pass

#    Search
./build/search_index \
  --index tmp/sift_index.bin \
  --data tmp/sift_base.fbin \
  --queries tmp/sift_query.fbin \
  --gt tmp/sift_gt.ibin \
  --K 10 --L 10,20,30,50,75,100,150,200
```

### Build Flags

| Flag | Default | Description |
|---|---|---|
| `--R` | 32 | Maximum out-degree per node |
| `--L` | 75 | Build-time search list size |
| `--alpha` | 1.2 | RNG pruning parameter (α > 1 keeps long-range edges) |
| `--gamma` | 1.5 | Degree multiplier for backward-edge pruning threshold |
| `--entry_points` | 1 | Number of k-means entry points |
| `--single_pass` | off | Skip two-pass build (faster, slightly lower quality) |
| `--soft_prune` | off | Use soft diversity pruning instead of hard α-RNG |
| `--soft_beta` | 0.5 | Diversity penalty strength (only with `--soft_prune`) |
| `--baseline` | off | Unimproved reference build: random start node, empty initial graph, RobustPrune over the top-L list |

CMake option `-DGRAPHANN_SIMD_REDUCTION=ON` lets the compiler reorder the float sum in the L2 kernels so it keeps vector accumulators. It is off by default because it changes floating-point summation order; results match to within rounding.

### Search Flags

| Flag | Description |
|---|---|
| `--quantized` | Use uint8 ADC for traversal, float32 re-ranking |
| `--dynamic` | Enable adaptive beam width |
| `--pca` | Use PCA-projected distances for traversal |
| `--pca_dim N` | PCA projection dimensionality (default 32) |

### Reproducing the benchmarks

```bash
# expects the texmex files in SIFT1M/ (sift_base.fvecs, sift_query.fvecs, sift_groundtruth.ivecs)
./scripts/benchmark_sift1m.sh
```

Builds the baseline, R=32 and R=64 indexes. Runs every search mode three times single-threaded (`OMP_NUM_THREADS=1`), and writes the tables in `results/sift1m.md`. If PyTorch can see a CUDA GPU, it also runs the GPU tools.

### GPU search

```bash
pip install torch numpy
python gpu/batched_search.py --index tmp/idx_impr_r64.bin --data tmp/sift_base.fbin \
    --queries tmp/sift_query.fbin --gt tmp/sift_gt.ibin --L 10,20,30,50,75,100 --batch 1024 --width 4
python gpu/bruteforce_knn.py --data tmp/sift_base.fbin --queries tmp/sift_query.fbin --K 10 --check tmp/sift_gt.ibin
```

On a laptop NVIDIA GPU with the R=64 index (`--batch 1024 --width 4`):

| L | Recall@10 | QPS |
|---|---|---|
| 10 | 0.9253 | 40,628 |
| 20 | 0.9695 | 33,125 |
| 50 | 0.9939 | 18,700 |
| 100 | 0.9983 | 12,434 |

See [gpu/README.md](gpu/README.md) for the knobs and how this compares with the CPU.

### HNSW Usage

```bash
# Build HNSW index
./build/build_hnsw \
  --data tmp/sift_base.fbin \
  --output tmp/hnsw_index.bin \
  --M 16 --efC 200

# Search HNSW index
./build/search_hnsw \
  --index tmp/hnsw_index.bin \
  --data tmp/sift_base.fbin \
  --queries tmp/sift_query.fbin \
  --gt tmp/sift_gt.ibin \
  --K 10 --ef 10,20,50,100,200,400
```

## Algorithm

### Build Phase
For each point (in random order, parallelized with OpenMP; each node's adjacency list is guarded by its own lock, including while RobustPrune rewrites it):
1. **Greedy Search** the current graph for the point, producing a candidate list of size `L`
2. **Robust Prune (α-RNG)** candidates to at most `R` diverse neighbors
3. **Add Edges** (forward + backward); prune any neighbor exceeding `γR` degree

### Search Phase
Greedy beam search from a fixed start node, maintaining a candidate set bounded at size `L`. Returns the top-`K` closest points.

```
  ┌──────────┐    ┌─────────────┐    ┌────────────┐    ┌──────────┐
  │ Load Data │───▶│ Build Graph │───▶│ Save Index │───▶│  Search  │
  └──────────┘    └─────────────┘    └────────────┘    └──────────┘
     fbin file       greedy_search       .bin file       beam search
                     robust_prune                        + re-ranking
```

---

## Implemented Improvements

### Build-Time Improvements (DiskANN Section 2)

| Improvement | Description | Measured effect (SIFT1M) |
|---|---|---|
| **Medoid Initialization** | Start from the point nearest the centroid instead of a random node | Part of the combined R=32 result below |
| **Random R-Regular Init** | Pre-seed the graph with R random neighbors per node | Early insertions have real search paths instead of an empty graph |
| **Full Visited Set → RobustPrune** | Pass every visited node, not just the top-L, to pruning | Better long-range edges (as in the paper) |
| **Two-Pass Build** | Pass 1 at α=1.0 (local), Pass 2 at α=1.2 (long-range) | About 2x build time; the benchmarks use `--single_pass` |
| **Higher Degree (R=64)** | Double the max out-degree | Average degree 75.8 vs 38.8; about 2.8x build time (134 s vs 47 s on 24 threads) |

Same beam width, exact float32 search:

| L | Baseline R=32 recall | Improved R=32 recall | Improved R=64 recall | Baseline / R=64 mean latency |
|---|---|---|---|---|
| 10 | 0.7720 | 0.8010 | **0.8865** | 105 / 132 µs |
| 20 | 0.8876 | 0.8992 | **0.9567** | 136 / 192 µs |
| 50 | 0.9648 | 0.9682 | **0.9916** | 228 / 349 µs |
| 75 | 0.9800 | 0.9823 | **0.9963** | 301 / 469 µs |
| 200 | 0.9952 | 0.9960 | **0.9991** | 661 / 1,025 µs |

At equal recall the R=64 graph needs a much smaller beam, which is where the latency gain in Key Results comes from.

### Search-Time Optimizations

**Flat sorted candidate list**: replaced `std::set` with a sorted `std::vector`. For L ≤ 200 the list fits in L1 cache (~1.6 KB), so an O(L) memmove beats a heap allocation per insert. Used in every configuration.

**Early-abandoning distance**: stops a distance computation once the partial sum exceeds the current worst candidate, checked every 16 dimensions. It is used in search only, never during the build, because RobustPrune needs exact distances.

**Vectorised reduction (`GRAPHANN_SIMD_REDUCTION`)**: without `-ffast-math`, GCC vectorises the subtract and multiply in the L2 loop but must add the 128 squares in order, which leaves a serial chain of scalar adds on the critical path. An `omp simd` reduction gives it 8-wide accumulators (a single `vfmadd231ps` loop). Same distance-computation counts and recall; about 20% lower latency (17-26%).

**Asymmetric Distance Computation (ADC)**: compresses the dataset from float32 (488 MiB) to uint8 (122 MiB, 4.0x). Traversal uses a float32 query against uint8 data; the final candidates are re-ranked exactly. See Negative Results.

**Dynamic beam width**: adaptive L that starts small and grows or shrinks with search progress. See Negative Results.

### Engineering fixes in the parallel build

* `robust_prune` rewrites a node's adjacency list under that node's lock, and the backward-edge loop iterates over a snapshot. Previously, concurrent searches could copy a list while it was being reassigned.
* `greedy_search` takes per-node locks, and skips the search-only shortcuts (early termination, early abandoning), whenever a build is running. Previously the second pass of the two-pass build read the graph without locks. That race is the most likely cause of the earlier "bus error" on R=64 two-pass builds.

---

### HNSW Comparison

A full **Hierarchical Navigable Small World (HNSW)** index is included for comparison; see [RESULTS.txt](RESULTS.txt) for its tables. Those HNSW and Vamana numbers come from an earlier run on a different machine, so compare them within that file only. The HNSW search allocates a fresh visited array for every layer of every query, which inflates its per-query cost relative to Vamana.

| Algorithm | Config | Recall@10 | Avg Latency |
|---|---|---|---|
| HNSW | M=16, efC=40, ef=100 | 0.9712 | 1228.1µs |
| Vamana | R=32, L=75 | 0.9828 | 898.1µs |
| Vamana | R=64 ADC, L=50 | 0.9936 | 1246.8µs |

---

### Negative Results

| Experiment | Outcome | Reason |
|---|---|---|
| ADC (uint8 traversal) | 17-44% slower than exact float32 at every L | De-quantising inside the inner loop (`q[i]*scale[i]+min[i]`) costs more than the memory traffic it saves without hand-written SIMD |
| Dynamic beam width (with ADC) | Within ±2% of plain ADC, no gain | Branching in a very tight loop; SIFT rarely has long "highways" that would justify shrinking L |
| Multiple entry points (k=8) | No benefit | SIFT1M lacks strong cluster structure |
| Strict γ removal (γ=1.0) | -0.02 recall | Too aggressive pruning |
| PCA traversal (32-dim) | 0.50 recall | SIFT has high intrinsic dimensionality |
| PCA traversal (64-dim) | 0.84 recall | Better but still 0.14 below exact |
| Quantization-aware refinement | -0.006 recall | Re-pruning without new candidates makes graph sparser |

---

### Above-and-Beyond

- **Hard Query Characterization** (`hard_query_analysis`): finds queries that fail at high L and correlates them with spatial properties
- **Degree Distribution Analysis**: export and visualise how build strategies change graph structure
- **Ablation Framework** (`run_ablation.sh`): every combination of baseline/improved build, single/two-pass and R=32/64
- **Soft Diversity Pruning**: alternative to hard α-RNG that scores candidates by a directional-overlap penalty
- **GPU tools** (`gpu/`): batched beam search over the built graph, plus exact brute-force k-NN, in PyTorch

---

## References

- Subramanya et al., *DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node*, NeurIPS 2019
- Malkov & Yashunin, *Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs*, IEEE TPAMI 2020

## License

Academic project for DA2303, IIT Madras, 2026.

