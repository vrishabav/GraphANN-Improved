# GraphANN — Vamana Index for Approximate Nearest Neighbor Search

**DA2303 — Algorithms for Data Science, Milestone 2**

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
graphann-main-milestone-2/
├── CMakeLists.txt              # Build system (3 executables + static lib)
├── README.md                   # This file
├── REPORT.md                   # Comprehensive project report
├── include/
│   ├── distance.h              # 5 distance functions (L2², ADC, PCA)
│   ├── io_utils.h              # fbin/ibin file I/O + aligned memory
│   ├── timer.h                 # Chrono stopwatch
│   └── vamana_index.h          # VamanaIndex class + GraphStats
├── src/
│   ├── distance.cpp            # Distance implementations
│   ├── io_utils.cpp            # File loader implementations
│   ├── vamana_index.cpp        # Core: greedy_search, robust_prune, build
│   ├── build_index.cpp         # CLI: build index from data file
│   ├── search_index.cpp        # CLI: search + recall/latency evaluation
│   └── hard_query_analysis.cpp # CLI: hard query characterization tool
└── scripts/
    ├── convert_vecs.py         # fvecs/ivecs → fbin/ibin converter
    ├── run_sift1m.sh           # End-to-end pipeline script
    ├── run_ablation.sh         # 16-condition ablation study
    └── analyze_results.py      # Plot generation (Pareto, histograms)
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

### Search Flags

| Flag | Description |
|---|---|
| `--quantized` | Use uint8 ADC for traversal, float32 re-ranking |
| `--dynamic` | Enable adaptive beam width (Proposal C) |
| `--pca` | Use PCA-projected distances for traversal |
| `--pca_dim N` | PCA projection dimensionality (default 32) |

---

## Implemented Improvements

### Search-Time Optimizations (Proposals A–D)

| Proposal | Description | Impact |
|---|---|---|
| **A: Scalar Quantization (ADC)** | uint8 dataset, float32 query, 4× memory reduction | -20% latency, -0.003 recall |
| **B: Early Abandoning** | Abort distance computation when partial sum > threshold | -12% latency when combined with ADC |
| **C: Dynamic Beam Width** | Adaptive L that shrinks on highways, expands when stuck | +5% latency, +0.002 recall |
| **D: Flat Sorted Vector** | Replace `std::set` with sorted `std::vector` for candidates | -28% latency, zero recall change |

### Build-Time Improvements (DiskANN Section 2)

| Improvement | Description | Impact |
|---|---|---|
| **Medoid Initialization** | Start search from geometric center instead of random node | +0.017 recall at L=10, -60% P99 |
| **Random R-Regular Init** | Pre-seed graph with R random neighbors per node | Eliminates cold-start problem |
| **Full Visited Set to RobustPrune** | Pass ALL visited nodes (not just top-L) to pruning | Better long-range edge selection |
| **Two-Pass Build** | Pass 1 at α=1.0 (local), Pass 2 at α=1.2 (long-range) | Higher quality but 2× slower |
| **Higher Degree (R=64)** | Double max out-degree from 32 to 64 | +0.087 recall at L=10, 3× build time |

### Additional Experiments (Negative Results)

| Experiment | Outcome | Reason |
|---|---|---|
| Multiple entry points (k=8) | No benefit | SIFT1M lacks strong cluster structure |
| Strict γ removal (γ=1.0) | -0.02 recall | Too aggressive pruning |
| PCA traversal (32-dim) | 0.50 recall | SIFT has high intrinsic dimensionality |
| PCA traversal (64-dim) | 0.84 recall | Better but still 0.14 below exact |
| Quantization-aware refinement | -0.006 recall | Re-pruning without new candidates makes graph sparser |
| Two-pass with R=64 | Bus error | Memory pressure during dense Pass 2 |

### Above-and-Beyond Analysis

- **Hard Query Characterization**: Tool to identify queries that fail at high L, correlate with spatial properties, and verify if build improvements fix them
- **Degree Distribution Analysis**: Export and visualize how build strategies change graph structure
- **Ablation Framework**: Systematic 16-condition study of all build improvement combinations

---

## License

Academic project for DA2303, IIT Madras, April 2026.
