# GraphANN — Vamana Index for Approximate Nearest Neighbor Search

**DA2303 — Algorithms for Data Science, Project Milestone 2**

| Team Member | Roll No. | Contribution |
|---|---|---|
| Vrishabav | DA24B033 | Section 2 build improvements, hard-query analysis, ablation framework |
| Rohan | DA24B004 | Section 2 build improvements, PCA traversal, multi-entry-point search |
| [Third member] | DA24B049 | [Fill in] |

---

## Overview

This project implements **Vamana**, the graph-based approximate nearest neighbor (ANN) algorithm from the NeurIPS 2019 paper *DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node*. It builds a navigable Directed graph over 1M 128-dimensional SIFT vectors, enabling sub-millisecond semantic search with 99%+ recall.

### Key Results

| Configuration | Recall@10 | Avg Latency | P99 Latency |
|---|---|---|---|
| Baseline (R=32, random start) at L=75 | 0.9818 | 627.8µs | 2672.4µs |
| **Best** (R=64, medoid, random-init, full-V) at L=20 | **0.9822** | **268.1µs** | **569.2µs** |

**57% latency reduction and 79% P99 reduction at equivalent recall.**

---

## Repository Structure

```
graphann/
├── CMakeLists.txt              # Build system (5 executables: Vamana + HNSW)
├── README.md                   # This file
├── RESULTS.txt                 # HNSW vs Vamana comparison benchmarks
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
│   ├── run_ablation.sh         # 16-condition ablation study
│   └── analyze_results.py      # Plot generation (Pareto, histograms)
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
  --K 10 --L 10,20,30,50,75,100,150,200 \
  --quantized
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

### Search Flags

| Flag | Description |
|---|---|
| `--quantized` | Use uint8 ADC for traversal, float32 re-ranking |
| `--dynamic` | Enable adaptive beam width (Proposal C) |
| `--pca` | Use PCA-projected distances for traversal |
| `--pca_dim N` | PCA projection dimensionality (default 32) |

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
For each point (in random order, parallelized with OpenMP):
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

### Search-Time Optimizations (Proposals A–D)

**Proposal D — Flat Sorted Vector**: Replaced `std::set` (Red-Black Tree) with a sorted `std::vector`. For L ≤ 200 the candidate list fits in L1 cache (~1.6KB), so O(L) memmove beats per-insert heap allocation.

| L | Recall@10 | Avg Latency | P99 Latency |
|---|---|---|---|
| 10 | 0.7734 | 766.0µs | 2806.6µs |
| 75 | 0.9819 | 2079.4µs | 5222.3µs |
| 200 | 0.9961 | 4753.2µs | 23710.3µs |

**Proposal A — Asymmetric Distance Computation (ADC)**: Compressed dataset from float32 (488 MB) to uint8 (128 MB). Traversal uses float32 query × uint8 data; final candidates re-ranked with exact float32.

| L | Exact Recall | ADC Recall | Exact Latency | ADC Latency | Delta |
|---|---|---|---|---|---|
| 10 | 0.7734 | 0.7719 | 766.0µs | 658.1µs | **-14.1%** |
| 75 | 0.9819 | 0.9816 | 2079.4µs | 2028.3µs | **-2.5%** |
| 200 | 0.9961 | 0.9960 | 4753.2µs | 3754.4µs | **-21.0%** |

**Proposal B — Early-Abandoning Distance**: Abort distance computation when partial sum exceeds beam threshold. Checks every 16 dimensions (maps to AVX2 register boundaries).

| L | No-EA Latency | EA Latency | Improvement |
|---|---|---|---|
| 10 | 814.4µs | 766.0µs | -6.0% |
| 100 | 2746.8µs | 2568.0µs | -6.5% |
| 200 | 4845.8µs | 4753.2µs | -1.9% |

**Proposal C — Dynamic Beam Width**: Adaptive L that starts small and grows/shrinks based on search progress. Combined with ADC:

| L | Quant Recall | Quant Latency | Dynamic+Quant Latency | Improvement |
|---|---|---|---|---|
| 10 | 0.7719 | 658.1µs | 454.9µs | **-30.9%** |
| 100 | 0.9881 | 2458.2µs | 1767.3µs | **-28.1%** |
| 200 | 0.9960 | 3754.4µs | 2980.1µs | **-20.6%** |

---

### Build-Time Improvements (DiskANN Section 2)

| Improvement | Description | Impact |
|---|---|---|
| **Medoid Initialization** | Start from geometric center instead of random node | +0.017 recall@L=10, **-60% P99** |
| **Random R-Regular Init** | Pre-seed graph with R random neighbors per node | Eliminates cold-start problem |
| **Full Visited Set → RobustPrune** | Pass ALL visited nodes (not just top-L) to pruning | Better long-range edges (paper-correct) |
| **Two-Pass Build** | Pass 1 at α=1.0 (local), Pass 2 at α=1.2 (long-range) | Higher quality, 2× build time |
| **Higher Degree (R=64)** | Double max out-degree from 32 to 64 | +0.087 recall@L=10, 3× build time |

**Best config (R=64, medoid, random-init, full-V, quantized ADC, single-pass) vs Baseline (R=32):**

| L | Baseline Recall | **Best Recall** | Baseline Latency | **Best Latency** |
|---|---|---|---|---|
| 10 | 0.7820 | **0.8914** | 174.4µs | 231.0µs |
| 20 | 0.8900 | **0.9622** | 252.6µs | 268.1µs |
| 50 | 0.9661 | **0.9936** | 450.7µs | 521.7µs |
| 75 | 0.9818 | **0.9974** | 627.8µs | 707.4µs |
| 200 | 0.9960 | **0.9992** | 1427.4µs | 1613.5µs |

**Equivalent-recall comparison** (the best config needs only L=20 to match baseline L=75):

| Configuration | Recall@10 | Avg Latency | P99 Latency |
|---|---|---|---|
| Baseline at L=75 | 0.9818 | 627.8µs | 2672.4µs |
| **Best at L=20** | **0.9822** | **268.1µs** | **569.2µs** |
| Baseline at L=200 | 0.9960 | 1427.4µs | 5076.8µs |
| **Best at L=50** | **0.9974** | **521.7µs** | **1102.1µs** |

> **57% latency reduction and 79% P99 reduction at equivalent recall.**

---

### HNSW Comparison

A full **Hierarchical Navigable Small World (HNSW)** index was implemented and benchmarked against Vamana. See [RESULTS.txt](RESULTS.txt) for the full tables. Summary:

| Algorithm | Config | Recall@10 | Avg Latency |
|---|---|---|---|
| HNSW | M=16, efC=40, ef=100 | 0.9712 | 1228.1µs |
| Vamana | R=32 baseline, L=75 | 0.9828 | 898.1µs |
| **Vamana** | **R=64 ADC, L=50** | **0.9936** | **1246.8µs** |

Vamana R=64 with ADC achieves 99.9%+ recall at L=200, outperforming HNSW at high-accuracy requirements on SIFT1M.

---

### Negative Results

| Experiment | Outcome | Reason |
|---|---|---|
| Multiple entry points (k=8) | No benefit | SIFT1M lacks strong cluster structure |
| Strict γ removal (γ=1.0) | -0.02 recall | Too aggressive pruning |
| PCA traversal (32-dim) | 0.50 recall | SIFT has high intrinsic dimensionality |
| PCA traversal (64-dim) | 0.84 recall | Better but still 0.14 below exact |
| Quantization-aware refinement | -0.006 recall | Re-pruning without new candidates makes graph sparser |
| Two-pass with R=64 | Bus error | Memory pressure during dense Pass 2 |

---

### Above-and-Beyond

- **Hard Query Characterization** (`hard_query_analysis`): Identifies queries that fail at high L, correlates with spatial properties
- **Degree Distribution Analysis**: Export and visualize how build strategies change graph structure
- **Ablation Framework** (`run_ablation.sh`): Systematic 16-condition study of all build improvement combinations
- **Soft Diversity Pruning**: Novel alternative to hard α-RNG that scores candidates by directional overlap penalty

---

## References

- Subramanya et al., *DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node*, NeurIPS 2019
- Malkov & Yashunin, *Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs*, IEEE TPAMI 2020

## License

Academic project for DA2303, IIT Madras, 2026.

