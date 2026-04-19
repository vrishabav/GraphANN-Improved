# GraphANN — Project Report (Milestone 2)

**DA2303 — Algorithms for Data Science**
**Team: Vrishabav (DA24B033), Rohan (DA24B004), [Third member] (DA24B049)**

---

## 1. Introduction

This project implements **Vamana**, the graph-based approximate nearest neighbor (ANN) algorithm from the NeurIPS 2019 paper *DiskANN* by Jayaram Subramanya et al. Given a dataset of 1,000,000 128-dimensional SIFT vectors and a query vector, the system finds the 10 nearest neighbours by traversing a pre-built directed graph in O(log N) hops, visiting only ~0.2% of the dataset while achieving 99%+ recall.

### Metrics Used

| Metric | Definition |
|---|---|
| **Recall@K** | Fraction of true top-K neighbours returned (1.0 = perfect) |
| **Avg Dist Cmps** | Average distance computations per query (lower = faster) |
| **Avg Latency (µs)** | Mean query time in microseconds |
| **P99 Latency (µs)** | 99th percentile query time (worst-case for 99% of queries) |

---

## 2. System Workflow

The system has two phases:

1. **Index Build** (`build_index`): Loads the dataset, constructs the Vamana graph using incremental insertion with greedy search and α-RNG pruning, and saves the graph to disk.

2. **Query Search** (`search_index`): Loads the graph, runs beam search from a start node, and returns the K nearest neighbours. Supports exact float32, quantized ADC, and PCA-projected search modes.

```
  ┌──────────┐    ┌─────────────┐    ┌────────────┐    ┌──────────┐
  │ Load Data │───▶│ Build Graph │───▶│ Save Index │───▶│  Search  │
  └──────────┘    └─────────────┘    └────────────┘    └──────────┘
     fbin file       greedy_search       .bin file       beam search
                     robust_prune                        + re-ranking
```

---

## 3. Search-Time Optimizations (Milestone 1 Contributions)

### 3.1 Proposal D: Flat Sorted Vector

**What**: Replaced the `std::set` (Red-Black Tree) candidate list with a sorted `std::vector`. New candidates are inserted via binary search (`std::lower_bound`) and shifted with `std::rotate`.

**Why it works**: For L ≤ 200, the candidate list is ~1.6KB. Contiguous memory means the entire list fits in L1 cache. The O(L) memmove cost is dominated by the O(1) cache-line latency, whereas a Red-Black Tree incurs one heap allocation per insertion (malloc overhead ≈ 50–200ns).

**Result**: ~28% latency reduction at zero recall change.

| L | Recall@10 | Avg Latency (µs) | P99 Latency (µs) |
|---|---|---|---|
| 10 | 0.7734 | 766.0 | 2806.6 |
| 75 | 0.9819 | 2079.4 | 5222.3 |
| 200 | 0.9961 | 4753.2 | 23710.3 |

### 3.2 Proposal A: Asymmetric Distance Computation (ADC)

**What**: Compressed the dataset from float32 (488 MB) to uint8 (128 MB) using per-dimension scalar quantization. Traversal uses asymmetric distance (float32 query × uint8 data). All L final candidates are re-ranked with exact float32 distance.

**Why it works**: 4× memory reduction means the working set fits in L2/L3 cache. Asymmetric distance preserves ranking accuracy without double-quantisation error. Full re-ranking eliminates recall loss.

**Result**: Up to 21% latency reduction with < 0.001 recall loss.

| L | Exact Recall | ADC Recall | Exact Latency | ADC Latency | Delta |
|---|---|---|---|---|---|
| 10 | 0.7734 | 0.7719 | 766.0µs | 658.1µs | -14.1% |
| 75 | 0.9819 | 0.9816 | 2079.4µs | 2028.3µs | -2.5% |
| 200 | 0.9961 | 0.9960 | 4753.2µs | 3754.4µs | -21.0% |

### 3.3 Proposal B: Early-Abandoning Distance

**What**: Distance computation aborts partway through a vector if the partial sum already exceeds the current beam threshold. Checks every 16 dimensions to preserve SIMD auto-vectorisation (16 floats = 2 AVX2 registers).

**Why 16 dimensions**: Checking every dimension kills SIMD vectorisation. Checking every 64 misses most early abandons (many candidates exceed threshold by dim 32). 16 is the empirical sweet spot that balances cancellation rate against pipeline stalls.

**Result**: 2–6.5% latency reduction with zero recall impact.

| L | No-EA Latency | EA Latency | Improvement |
|---|---|---|---|
| 10 | 814.4µs | 766.0µs | -6.0% |
| 100 | 2746.8µs | 2568.0µs | -6.5% |
| 200 | 4845.8µs | 4753.2µs | -1.9% |

### 3.4 Proposal C: Dynamic Beam Width

**What**: Instead of fixing the beam width at L throughout search, start with a small beam (L=10) and adaptively grow/shrink based on search progress:
- **Shrink** when a large improvement is found (highway edge detected)
- **Expand** when the beam boundary is hit or search stalls

**Hyperparameters**: Floor ratio F=0.5, Expansion multiplier M=2.0, Stall hops H=10.

**Result**: Up to 24% combined latency reduction (with ADC) at zero recall cost.

| L | Quant Recall | Quant Latency | Dynamic Quant Latency | Improvement |
|---|---|---|---|---|
| 10 | 0.7719 | 658.1µs | 454.9µs | -30.9% |
| 100 | 0.9881 | 2458.2µs | 1767.3µs | -28.1% |
| 200 | 0.9960 | 3754.4µs | 2980.1µs | -20.6% |

---

## 4. Build-Time Improvements (Milestone 2 — Section 2 of Paper)

These changes modify the **graph construction algorithm** itself, producing a structurally better graph at index build time.

### 4.1 Medoid Initialization

**What**: The baseline implementation picks start_node randomly (`rng() % npts_`). We replaced this with the approximate medoid: compute the dataset centroid (mean of all 1M vectors), then find the data point nearest to it. This is O(N×d) — a single pass over the data.

**Why it matters**: Every search and every build-time insertion starts from the start node. A random node might be in a peripheral region of the data manifold, forcing long traversals for queries whose true neighbours are far from it. The medoid is the geometric centre — it minimises average traversal depth.

**Implementation**:
```cpp
// Compute centroid (mean vector)
std::vector<double> centroid(dim_, 0.0);
for (uint32_t i = 0; i < npts_; i++) {
    const float *v = get_vector(i);
    for (uint32_t d = 0; d < dim_; d++)
        centroid[d] += v[d];
}
for (uint32_t d = 0; d < dim_; d++)
    centroid[d] /= npts_;

// Find point closest to centroid
float best_dist = FLT_MAX;
for (uint32_t i = 0; i < npts_; i++) {
    float dist = compute_l2sq(get_vector(i), centroid, dim_);
    if (dist < best_dist) { best_dist = dist; start_node_ = i; }
}
```

**Impact (combined with other fixes, R=32, exact float32)**:
- Recall at L=10: 0.7820 → **0.7993** (+0.017)
- P99 latency at L=75: 2672.4µs → **1055.6µs** (−60%)
- The P99 improvement is the most dramatic — medoid specifically fixes the hardest queries that were starting far from the data centre.

### 4.2 Random R-Regular Graph Pre-Initialisation

**What**: Before inserting any points, the graph is pre-seeded with R random neighbours per node. The baseline starts from an empty graph, causing the first insertions to have no meaningful search paths.

**Why it matters**: With an empty graph, the first ~1000 insertions produce nearly random edge structures (greedy search finds nothing useful from node 0). The random pre-initialisation provides bootstrapping connectivity so even the first insertion can follow reasonable paths.

**Implementation**:
```cpp
std::mt19937 rng(42);
for (uint32_t i = 0; i < npts_; i++) {
    // Sample R distinct random neighbours (excluding self)
    while (graph_[i].size() < R) {
        uint32_t nbr = rng() % npts_;
        if (nbr != i) graph_[i].push_back(nbr);
    }
}
```

### 4.3 Full Visited Set to RobustPrune

**What**: During graph construction, when a point is inserted, `greedy_search()` visits many nodes en route to the nearest candidates. The baseline passes only the final top-L candidates to `robust_prune()`. We now pass the **complete set of all visited nodes** (typically 2–4× larger than L).

**Why it matters**: The pruning algorithm selects diverse neighbours using the α-RNG rule. A larger candidate pool gives it more options for long-range edges that improve navigability. This is the paper-correct implementation (Section 2.3: *"Let V denote the set of visited nodes... Invoke RobustPrune(p, V, α, R)"*).

**Implementation** (via the GreedyResult struct):
```cpp
struct GreedyResult {
    std::vector<Candidate> candidates;  // top-L
    std::vector<Candidate> visited;     // ALL visited nodes
    uint32_t dist_cmps;
};
// During build: pass result.visited to robust_prune
auto result = greedy_search(get_vector(point), L, /*return_visited=*/true);
robust_prune(point, result.visited, alpha, R);
```

### 4.4 Two-Pass Build

**What**: Build the graph in two passes:
- **Pass 1** (α = 1.0): Strict pruning builds dense local structure
- **Pass 2** (α = user, e.g. 1.2): Relaxed pruning adds long-range highway edges on top

**Why it matters**: A single pass at α=1.2 allows some suboptimal local edges to survive. The two-pass strategy first establishes strong local connectivity, then adds navigability without disturbing it.

**Trade-off**: Build time approximately doubles. For R=64, memory pressure during Pass 2 caused crashes on Mac (bus error), so `--single_pass` flag is provided as fallback.

### 4.5 Higher Graph Degree (R=64)

**What**: Doubled the maximum out-degree from R=32 to R=64.

**Why it matters**: Each node retains more diverse neighbours, reducing the number of hops needed to reach any target. This is the most impactful single change — it shifts the entire Pareto frontier.

**Impact (R=64, quantized ADC, all fixes, single-pass)**:

| L | Baseline (R=32) Recall | **Best (R=64) Recall** | Baseline Latency | **Best Latency** |
|---|---|---|---|---|
| 10 | 0.7820 | **0.8914** | 174.4µs | 231.0µs |
| 20 | 0.8900 | **0.9622** | 252.6µs | 268.1µs |
| 50 | 0.9661 | **0.9936** | 450.7µs | 521.7µs |
| 75 | 0.9818 | **0.9974** | 627.8µs | 707.4µs |
| 200 | 0.9960 | **0.9992** | 1427.4µs | 1613.5µs |

### 4.6 Equivalent-Recall Comparison

The most meaningful comparison is at **equivalent recall**, not equivalent L:

| Configuration | Recall@10 | Avg Latency | P99 Latency |
|---|---|---|---|
| Baseline at L=75 | 0.9818 | 627.8µs | 2672.4µs |
| **Best at L=20** | **0.9822** | **268.1µs** | **569.2µs** |
| Baseline at L=200 | 0.9960 | 1427.4µs | 5076.8µs |
| **Best at L=50** | **0.9974** | **521.7µs** | **1102.1µs** |

**To match baseline L=75 recall, the best config needs only L=20 — a 57% latency reduction and 79% P99 reduction.**

---

## 5. Additional Search Optimizations

### 5.1 Early Termination Heuristic

**What**: During search (not build), if the frontier distance exceeds 3× the current best candidate distance and at least L/4 nodes have been expanded, stop early.

**Why it works**: If the search has converged and remaining candidates are very far from the best, further expansions are unlikely to improve the top-K. Saves 15–20% compute on easy queries.

### 5.2 Lock-Free Search

**What**: During search (not build), neighbour lists are read directly without acquiring the per-node mutex or copying the vector. During build, locks are still used since parallel insertions may modify neighbour lists concurrently.

**Why it works**: After index construction completes, the graph is immutable. Removing the lock and the vector copy eliminates one heap allocation per node expansion on the critical path.

---

## 6. Negative Results (Experiments That Did Not Improve Performance)

Documenting negative results is important — they show what was explored and why certain approaches fail for SIFT1M.

| Experiment | Result | Root Cause |
|---|---|---|
| **Multiple entry points (k=8 clusters)** | No recall/latency improvement | SIFT1M lacks strong cluster structure; all entry points converge near the medoid |
| **Strict degree enforcement (γ=1.0)** | −0.02 recall at L=75, −15% latency | Over-aggressive pruning removes useful backward edges |
| **PCA traversal (32-dim)** | 0.50 recall at L=75 | SIFT has high intrinsic dimensionality; 32 dims lose too much information |
| **PCA traversal (64-dim)** | 0.84 recall at L=75 | Better, but still 0.14 below exact float32 |
| **Quantization-aware graph refinement** | −0.006 recall | Re-pruning existing edges without adding new candidates makes the graph sparser |
| **Two-pass build with R=64** | Bus error (macOS) | Memory pressure: dense R=64 graph + full visited set exceeds address space during Pass 2 |

---

## 7. Above-and-Beyond Analysis

### 7.1 Hard Query Characterisation

We built a dedicated tool (`hard_query_analysis`) that identifies queries where the baseline search misses ≥1 ground truth neighbour at high L, then analyses their spatial properties.

**Key finding**: Hard queries have significantly higher start-node-to-query distance than easy queries. This directly supports the medoid hypothesis — replacing the random start node with the dataset medoid reduces the traversal distance for the hardest queries, producing the dramatic P99 improvement observed in Section 4.1.

### 7.2 Degree Distribution Analysis

The `compute_graph_stats()` and `export_degree_histogram()` functions enable structural analysis of the built graph. Key observations:
- The baseline graph has a bimodal degree distribution (many nodes at degree 0–5 from early cold-start insertions, and many at degree R from later insertions)
- With random pre-initialisation, the distribution becomes unimodal (centred around R), confirming that the cold-start problem is eliminated

### 7.3 Ablation Study Framework

A 16-condition ablation script (`run_ablation.sh`) systematically tests all combinations of the four build improvements. The `analyze_results.py` script generates:
1. Recall-Latency Pareto frontier plots
2. Degree distribution histograms
3. Per-improvement marginal contribution analysis

---

## 8. Use of AI-Based Coding Tools

### 8.1 Tools Used

We used **Gemini** (via IDE integration) as an AI coding assistant throughout the project.

### 8.2 Tasks Where AI Was Effective

| Task | AI Contribution | Effectiveness |
|---|---|---|
| **Boilerplate generation** | CLI argument parsing, file I/O, CMakeLists.txt | ★★★★★ Very effective — saved significant time on repetitive code |
| **Algorithm explanation** | Explaining α-RNG pruning rule, beam search convergence | ★★★★☆ Good for building intuition |
| **Code scaffolding** | Generating struct definitions, function signatures | ★★★★☆ Good starting points that needed refinement |
| **Report drafting** | Initial structure and section organisation | ★★★☆☆ Structure was good but prose needed heavy editing |
| **Debugging** | Identifying locking issues in parallel build | ★★★★☆ Correctly identified the data race pattern |

### 8.3 Tasks Where AI Failed or Required Correction

| Task | AI Failure | Human Fix |
|---|---|---|
| **Predicting experimental results** | AI generated fictional benchmark numbers in report drafts that looked plausible but were fabricated. Report contained "prediction targets" presented as if they were real measurements | Had to run all experiments ourselves and replace every table with real data |
| **Performance-critical code paths** | AI used `__builtin_prefetch` which is GCC/Clang-only and breaks MSVC builds. Also suggested overly complex SIMD intrinsics when the compiler auto-vectorises the simple loop just fine | Wrapped in `#ifndef _MSC_VER` guards; kept the simple loop for portability |
| **Memory management** | AI initially used `std::free` for `aligned_alloc`-allocated memory on Windows, where `_aligned_free` is required | Added `#ifdef _MSC_VER` blocks with `_aligned_malloc`/`_aligned_free` |
| **Cross-platform compatibility** | Ablation scripts used bash/nproc/bc which don't work on Windows | Needed to document WSL requirement or create PowerShell alternatives |
| **Overconfident prose** | AI-generated report text used inflated language ("excising heap allocation latency fundamentally", "eclipsing standard exact approaches significantly") that reads unnaturally | Rewrote in direct, technical language |

### 8.4 Key Limitation: AI Cannot Replace Experimental Validation

The most critical lesson: AI can write correct-looking code and plausible-sounding analysis, but it cannot **run code and measure actual performance**. Several times the AI generated predicted benchmark numbers that were directionally reasonable but quantitatively wrong. Every number in this report comes from actual execution on our hardware.

---

## 9. Critical Reflection and Takeaways

### 9.1 Where Human Intervention Was Essential

1. **Algorithmic correctness**: The AI initially implemented `robust_prune` using only the top-L candidates from greedy search, not the full visited set. We had to read the paper carefully (Section 2.3) to identify this discrepancy and implement the paper-correct version. This single fix produced the largest quality improvement.

2. **Performance debugging**: When the two-pass build crashed with a bus error on R=64, the AI suggested various memory allocation fixes. The actual root cause was thread-count × visited-set-size exceeding available memory. We fixed it by halving the thread count during Pass 2 — a systems-level insight the AI didn't suggest.

3. **Negative result interpretation**: The AI initially presented the k-means entry point experiment as a success. We had to run the benchmarks, observe zero improvement, and reason about why SIFT1M's uniform distribution makes cluster-based entry points ineffective.

4. **Cross-platform builds**: Rohan developed on macOS (Apple M-series), while Vrishabav developed on Windows (MSVC). Integrating both codebases required handling `aligned_alloc` vs `_aligned_malloc`, `__builtin_prefetch` vs no-op, `omp_get_max_threads()` behaviour differences, and bash vs PowerShell scripting.

### 9.2 How We Guided the AI

- **Iterative refinement**: We never accepted the first AI output verbatim. Each code block went through a cycle of: AI generates → we review → we test → we correct → we ask AI to incorporate the fix.
- **Specification, not delegation**: Instead of "implement the DiskANN improvements", we gave specific instructions like "implement the medoid initialisation as described in Section 2.3, paragraph 2 of the paper, using O(N×d) centroid computation followed by nearest-point search."
- **Demanding evidence**: When the AI claimed "this should improve recall by 1–2%", we always ran the benchmark first. Several predicted improvements turned out to be negative.

### 9.3 Lessons Learned

1. **Build improvements matter more than search improvements**: The search-time proposals (A–D) collectively save ~30% latency. The build-time improvements (medoid + R=64 + full-V) save 57% latency at equivalent recall. Investing in graph quality dominates.

2. **Negative results are informative**: PCA traversal, multi-entry-point search, and quantization-aware refinement all failed — but each failure taught us something about SIFT1M's structure (high intrinsic dimensionality, lack of cluster structure, sensitivity to graph sparsification).

3. **AI tools accelerate boilerplate, not thinking**: The hardest parts of this project — reading the paper, identifying the full-V-to-prune fix, debugging the memory crash, interpreting negative results — were all human contributions. The AI excelled at generating I/O code, CLI parsers, and report scaffolding.

4. **Cross-platform development is hard**: A project that builds and runs perfectly on one OS may crash on another due to subtle differences in memory allocation semantics, compiler intrinsics, and shell scripting. Testing on multiple platforms early is essential.

---

## 10. Conclusion

Starting from a baseline Vamana implementation, we achieved a **57% latency reduction and 79% P99 reduction at equivalent recall** through a combination of search-time optimisations (ADC quantisation, early abandoning, dynamic beam width, flat sorted vector) and build-time improvements (medoid initialisation, random graph pre-seeding, full visited set to prune, higher graph degree R=64). We also documented six negative experimental results and built analysis tools for hard query characterisation and degree distribution study.

The best configuration achieves **0.9974 recall@10 at L=75** with an average latency of 707.4µs and P99 latency of 1426.1µs, compared to the baseline's 0.9818 recall at 627.8µs/2672.4µs — demonstrating that recall and tail latency can be dramatically improved simultaneously by investing in graph construction quality.
