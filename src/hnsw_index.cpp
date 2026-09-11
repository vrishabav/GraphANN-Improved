#include "hnsw_index.h"
#include "distance.h"
#include "io_utils.h"
#include "timer.h"

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <omp.h>

#ifdef _WIN32
#include <malloc.h>
#define aligned_alloc(alignment, size) _aligned_malloc(size, alignment)
#define aligned_free(ptr) _aligned_free(ptr)
#else
#include <cstdlib>
#define aligned_free(ptr) free(ptr)
#endif

// ============================================================================
// Destructor
// ============================================================================

HNSWIndex::~HNSWIndex() {
    if (owns_data_ && data_) {
        aligned_free(data_);
        data_ = nullptr;
    }
}

// ============================================================================
// Random Level Assignment
// ============================================================================
// Each node is assigned a maximum layer drawn from a geometric distribution:
//   level = floor(-ln(uniform(0,1)) * mL)
// where mL = 1/ln(M). This ensures the expected number of nodes at layer l
// is N * exp(-l / mL), giving an exponentially thinning hierarchy.
// The maximum level is capped at a reasonable bound to prevent degenerate cases.

uint32_t HNSWIndex::random_level(std::mt19937& rng) const {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double r = dist(rng);
    // Avoid log(0)
    if (r < 1e-10) r = 1e-10;
    int level = static_cast<int>(std::floor(-std::log(r) * mL_));
    // Cap at a reasonable maximum to prevent memory explosion
    // For N=1M, M=16: expected max layer ≈ log(N)/log(M) ≈ 5
    const int MAX_LEVEL = 16;
    return static_cast<uint32_t>(std::min(level, MAX_LEVEL));
}

// ============================================================================
// Search Layer (Algorithm 2 from the paper)
// ============================================================================
// Beam search at a single layer of the HNSW graph.
// Starts from entry_point, maintains:
//   - candidates: min-heap by distance (closest first) — the "frontier"
//   - result:     max-heap by distance (farthest first) — top-ef found so far
// Terminates when the closest unvisited candidate is farther than the
// farthest node in the result set (no improvement possible).
//
// Returns: up to ef candidates sorted by distance ascending.

std::vector<HNSWIndex::Candidate>
HNSWIndex::search_layer(const float* query, uint32_t entry_point,
                         uint32_t ef, uint32_t layer) const {
    // visited: tracks which nodes we've computed distance for
    std::vector<bool> visited(npts_, false);

    float entry_dist = compute_l2sq(query, get_vector(entry_point), dim_);
    visited[entry_point] = true;

    // candidates: min-heap (closest at top) — nodes to expand
    // Use a vector + push_heap/pop_heap for efficiency
    std::vector<Candidate> candidates;
    candidates.reserve(ef + 1);
    candidates.push_back({entry_dist, entry_point});

    // result: max-heap (farthest at top) — best ef found so far
    std::vector<Candidate> result;
    result.reserve(ef + 1);
    result.push_back({entry_dist, entry_point});

    uint32_t dist_cmps = 1;

    while (!candidates.empty()) {
        // Pop closest candidate
        // candidates is maintained as a min-heap
        std::pop_heap(candidates.begin(), candidates.end(),
                      [](const Candidate& a, const Candidate& b) {
                          return a.first > b.first;  // min-heap: smallest first
                      });
        auto [c_dist, c_id] = candidates.back();
        candidates.pop_back();

        // Termination: if closest candidate is farther than worst in result,
        // no further improvement is possible
        float worst_result_dist = result.front().first;  // max-heap top = farthest
        if (c_dist > worst_result_dist && (uint32_t)result.size() >= ef)
            break;

        // Expand neighbors of c_id at this layer
        const std::vector<uint32_t>& neighbors = graph_[layer][c_id];

        for (uint32_t nbr : neighbors) {
            if (visited[nbr]) continue;
            visited[nbr] = true;

            float d = compute_l2sq(query, get_vector(nbr), dim_);
            dist_cmps++;

            float worst = result.front().first;
            if (d < worst || (uint32_t)result.size() < ef) {
                // Add to candidates (min-heap)
                candidates.push_back({d, nbr});
                std::push_heap(candidates.begin(), candidates.end(),
                               [](const Candidate& a, const Candidate& b) {
                                   return a.first > b.first;
                               });

                // Add to result (max-heap)
                result.push_back({d, nbr});
                std::push_heap(result.begin(), result.end());  // default: max-heap

                // Trim result to ef
                if ((uint32_t)result.size() > ef) {
                    std::pop_heap(result.begin(), result.end());
                    result.pop_back();
                }
            }
        }
    }

    // Sort result ascending by distance
    std::sort(result.begin(), result.end());
    return result;
}

// ============================================================================
// Heuristic Neighbor Selection (Algorithm 4 from the paper)
// ============================================================================
// Given a set of candidates for node node_id, select up to M diverse neighbors.
//
// The heuristic: process candidates in order of increasing distance to node_id.
// Accept candidate c if:
//   dist(node_id, c) < dist(c, s)  for ALL already-selected neighbors s
//
// This is equivalent to: c is closer to node_id than to any already-selected
// neighbor. It ensures the selected set "covers" different directions from
// node_id, preventing all neighbors from clustering in one region.
//
// extend_candidates: also consider neighbors-of-candidates as candidates
//   (improves recall at higher layers where the graph is sparse)
// keep_pruned: if we can't fill M slots with diverse candidates, fill
//   remaining slots with the best pruned candidates (maintains connectivity)

std::vector<uint32_t>
HNSWIndex::select_neighbors_heuristic(uint32_t node_id,
                                       std::vector<Candidate>& candidates,
                                       uint32_t M, uint32_t layer,
                                       bool extend_candidates,
                                       bool keep_pruned) const {
    // Optionally extend candidates with neighbors-of-candidates
    if (extend_candidates && layer < (uint32_t)graph_.size()) {
        std::unordered_set<uint32_t> seen;
        seen.reserve(candidates.size() * 2);
        for (auto& [d, id] : candidates) seen.insert(id);
        seen.insert(node_id);

        std::vector<Candidate> extra;
        for (auto& [d, id] : candidates) {
            if (layer < (uint32_t)graph_.size() && id < (uint32_t)graph_[layer].size()) {
                for (uint32_t nbr : graph_[layer][id]) {
                    if (seen.find(nbr) == seen.end()) {
                        seen.insert(nbr);
                        float nd = compute_l2sq(get_vector(node_id), get_vector(nbr), dim_);
                        extra.push_back({nd, nbr});
                    }
                }
            }
        }
        for (auto& c : extra) candidates.push_back(c);
    }

    // Sort candidates by distance to node_id (ascending)
    std::sort(candidates.begin(), candidates.end());

    // Remove node_id itself from candidates
    candidates.erase(
        std::remove_if(candidates.begin(), candidates.end(),
                       [node_id](const Candidate& c) { return c.second == node_id; }),
        candidates.end());

    std::vector<uint32_t> selected;
    selected.reserve(M);
    std::vector<Candidate> pruned;  // candidates that failed the diversity check

    for (auto& [dist_to_node, cand_id] : candidates) {
        if ((uint32_t)selected.size() >= M) break;

        // Check diversity: accept c if it's closer to node_id than to any
        // already-selected neighbor
        bool good = true;
        for (uint32_t s : selected) {
            float dist_cand_to_s = compute_l2sq(get_vector(cand_id), get_vector(s), dim_);
            if (dist_cand_to_s < dist_to_node) {
                // c is closer to s than to node_id — it's "covered" by s
                good = false;
                break;
            }
        }

        if (good) {
            selected.push_back(cand_id);
        } else if (keep_pruned) {
            pruned.push_back({dist_to_node, cand_id});
        }
    }

    // Fill remaining slots with pruned candidates (maintains connectivity)
    if (keep_pruned) {
        for (auto& [d, id] : pruned) {
            if ((uint32_t)selected.size() >= M) break;
            selected.push_back(id);
        }
    }

    return selected;
}

// ============================================================================
// Build (Algorithm 1 from the paper)
// ============================================================================
// Sequential insertion of all N points. For each new point q:
//   1. Assign random level l = random_level()
//   2. If l > current max_layer: update global entry point
//   3. Greedy descent from top layer to l+1 (ef=1 per layer)
//   4. Beam search from layer l down to 0 (ef=efConstruction)
//   5. At each layer lc <= l: select M (or M0 for lc=0) neighbors using
//      heuristic selection, add bidirectional edges, prune if over-degree
//
// Note: We use sequential insertion (not parallel) to maintain correctness.
// The HNSW paper does not describe a parallel build algorithm; parallel
// variants require careful locking and can degrade graph quality.
// For 1M points with M=16, efC=100, sequential build takes ~3-5 minutes.

void HNSWIndex::build(const std::string& data_path, uint32_t M,
                       uint32_t efConstruction, uint32_t seed) {
    M_              = M;
    M0_             = 2 * M;
    efConstruction_ = efConstruction;
    mL_             = 1.0 / std::log(static_cast<double>(M));

    // --- Load data ---
    std::cout << "Loading data from " << data_path << "..." << std::endl;
    FloatMatrix mat = load_fbin(data_path);
    npts_ = mat.npts;
    dim_  = mat.dims;

    // Allocate aligned memory for data
    size_t data_bytes = (size_t)npts_ * dim_ * sizeof(float);
    size_t aligned_bytes = (data_bytes + 63) & ~(size_t)63;
    data_ = static_cast<float*>(aligned_alloc(64, aligned_bytes));
    if (!data_) throw std::runtime_error("Failed to allocate data memory");
    std::memcpy(data_, mat.data.get(), data_bytes);
    owns_data_ = true;

    std::cout << "  Points: " << npts_ << ", Dimensions: " << dim_ << std::endl;
    std::cout << "  M=" << M_ << ", M0=" << M0_
              << ", efConstruction=" << efConstruction_
              << ", mL=" << mL_ << std::endl;

    // --- Initialize graph structure ---
    // We pre-allocate layer 0 for all nodes. Higher layers are added as needed.
    // graph_[layer] is a vector of size npts_, where graph_[layer][i] is the
    // neighbor list of node i at that layer.
    // We start with just layer 0.
    const uint32_t MAX_LAYERS = 17;  // 16 + layer 0
    graph_.resize(MAX_LAYERS);
    for (uint32_t l = 0; l < MAX_LAYERS; l++)
        graph_[l].resize(npts_);

    node_level_.resize(npts_, 0);
    locks_ = std::vector<std::mutex>(npts_);

    // --- Sequential insertion ---
    std::mt19937 rng(seed);
    max_layer_   = 0;
    entry_point_ = 0;

    // Insert first point manually (no search needed)
    node_level_[0] = random_level(rng);
    max_layer_     = node_level_[0];
    entry_point_   = 0;

    Timer build_timer;

    for (uint32_t q = 1; q < npts_; q++) {
        if (q % 50000 == 0) {
            std::cout << "\r  Inserted " << q << " / " << npts_
                      << " (max_layer=" << max_layer_ << ")" << std::flush;
        }

        const float* qvec = get_vector(q);
        uint32_t q_level  = random_level(rng);
        node_level_[q]    = q_level;

        uint32_t ep = entry_point_;

        // --- Phase 1: Greedy descent from max_layer to q_level+1 ---
        for (int lc = (int)max_layer_; lc > (int)q_level; lc--) {
            auto result = search_layer(qvec, ep, 1, (uint32_t)lc);
            if (!result.empty())
                ep = result[0].second;
        }

        // --- Phase 2: Beam search from q_level down to 0 ---
        for (int lc = (int)std::min(q_level, max_layer_); lc >= 0; lc--) {
            uint32_t layer = (uint32_t)lc;
            uint32_t Mmax  = (layer == 0) ? M0_ : M_;

            auto candidates = search_layer(qvec, ep, efConstruction_, layer);
            if (!candidates.empty())
                ep = candidates[0].second;

            auto neighbors = select_neighbors_heuristic(q, candidates, Mmax, layer, false, true);
            graph_[layer][q] = neighbors;

            for (uint32_t n : neighbors) {
                // Bidirectional edges still need locks if we ever go back to parallel,
                // but in sequential it's safe without them. Keeping for consistency.
                std::lock_guard<std::mutex> lock(locks_[n]);
                graph_[layer][n].push_back(q);

                if ((uint32_t)graph_[layer][n].size() > Mmax) {
                    std::vector<Candidate> n_cands;
                    n_cands.reserve(graph_[layer][n].size());
                    for (uint32_t nn : graph_[layer][n]) {
                        float d = compute_l2sq(get_vector(n), get_vector(nn), dim_);
                        n_cands.push_back({d, nn});
                    }
                    auto pruned = select_neighbors_heuristic(n, n_cands, Mmax, layer, false, true);
                    graph_[layer][n] = pruned;
                }
            }
        }

        // --- Update global entry point if q has a higher layer ---
        if (q_level > max_layer_) {
            max_layer_   = q_level;
            entry_point_ = q;
        }
    }

    double build_time = build_timer.elapsed_seconds();
    std::cout << "\nBuild complete in " << build_time << " seconds." << std::endl;
    std::cout << "Max layer: " << max_layer_ << std::endl;
    std::cout << "Entry point: " << entry_point_ << std::endl;

    // Print layer statistics
    auto avg_degs = layer_avg_degrees();
    for (uint32_t l = 0; l <= max_layer_; l++) {
        std::cout << "  Layer " << l << ": avg_degree=" << avg_degs[l] << std::endl;
    }
}

// ============================================================================
// Search (Algorithm 5 from the paper)
// ============================================================================
// 1. Greedy descent from max_layer to layer 1 (ef=1 per layer)
// 2. Beam search at layer 0 with ef=efSearch
// 3. Return top-K from layer 0 results

HNSWSearchResult HNSWIndex::search(const float* query, uint32_t K, uint32_t ef) const {
    if (ef < K) ef = K;

    Timer t;
    uint32_t total_dist_cmps = 0;

    uint32_t ep = entry_point_;

    // Phase 1: Greedy descent from max_layer to layer 1
    for (int lc = (int)max_layer_; lc > 0; lc--) {
        auto result = search_layer(query, ep, 1, (uint32_t)lc);
        if (!result.empty())
            ep = result[0].second;
        // Approximate dist_cmps for upper layers (small, not tracked precisely)
        total_dist_cmps += (uint32_t)graph_[(uint32_t)lc][ep].size() + 1;
    }

    // Phase 2: Beam search at layer 0
    auto candidates = search_layer(query, ep, ef, 0);

    double latency = t.elapsed_us();

    // Count distance computations at layer 0 (approximate)
    // The search_layer function doesn't return dist_cmps, so we estimate
    // based on ef and average degree at layer 0
    total_dist_cmps += (uint32_t)candidates.size() * 2;  // rough estimate

    HNSWSearchResult result;
    result.latency_us = latency;
    result.dist_cmps  = total_dist_cmps;
    result.ids.reserve(K);
    for (uint32_t i = 0; i < K && i < (uint32_t)candidates.size(); i++)
        result.ids.push_back(candidates[i].second);

    return result;
}

// ============================================================================
// Layer Average Degrees
// ============================================================================

std::vector<double> HNSWIndex::layer_avg_degrees() const {
    uint32_t num_layers = (uint32_t)graph_.size();
    std::vector<double> avg_degs(num_layers, 0.0);

    for (uint32_t l = 0; l < num_layers; l++) {
        if (graph_[l].empty()) continue;
        uint32_t count = 0;
        size_t total_edges = 0;
        for (uint32_t i = 0; i < npts_; i++) {
            if (node_level_[i] >= l) {
                total_edges += graph_[l][i].size();
                count++;
            }
        }
        avg_degs[l] = (count > 0) ? (double)total_edges / count : 0.0;
    }
    return avg_degs;
}

// ============================================================================
// Save / Load
// ============================================================================
// Binary format:
//   [uint32] npts
//   [uint32] dim
//   [uint32] M
//   [uint32] M0
//   [uint32] efConstruction
//   [uint32] max_layer
//   [uint32] entry_point
//   [double] mL
//   [uint32 * npts] node_level
//   For each layer l in [0, max_layer]:
//     For each node i in [0, npts):
//       [uint32] degree at layer l
//       [uint32 * degree] neighbor IDs

void HNSWIndex::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open())
        throw std::runtime_error("Cannot open file for writing: " + path);

    out.write(reinterpret_cast<const char*>(&npts_),          4);
    out.write(reinterpret_cast<const char*>(&dim_),           4);
    out.write(reinterpret_cast<const char*>(&M_),             4);
    out.write(reinterpret_cast<const char*>(&M0_),            4);
    out.write(reinterpret_cast<const char*>(&efConstruction_),4);
    out.write(reinterpret_cast<const char*>(&max_layer_),     4);
    out.write(reinterpret_cast<const char*>(&entry_point_),   4);
    out.write(reinterpret_cast<const char*>(&mL_),            sizeof(double));

    // Node levels
    out.write(reinterpret_cast<const char*>(node_level_.data()),
              npts_ * sizeof(uint32_t));

    // Graph layers 0..max_layer
    for (uint32_t l = 0; l <= max_layer_; l++) {
        for (uint32_t i = 0; i < npts_; i++) {
            uint32_t deg = (node_level_[i] >= l)
                           ? (uint32_t)graph_[l][i].size()
                           : 0;
            out.write(reinterpret_cast<const char*>(&deg), 4);
            if (deg > 0)
                out.write(reinterpret_cast<const char*>(graph_[l][i].data()),
                          deg * sizeof(uint32_t));
        }
    }

    std::cout << "HNSW index saved to " << path << std::endl;
}

void HNSWIndex::load(const std::string& index_path,
                      const std::string& data_path) {
    // Load data vectors
    FloatMatrix mat = load_fbin(data_path);
    npts_ = mat.npts;
    dim_  = mat.dims;

    size_t data_bytes    = (size_t)npts_ * dim_ * sizeof(float);
    size_t aligned_bytes = (data_bytes + 63) & ~(size_t)63;
    data_ = static_cast<float*>(aligned_alloc(64, aligned_bytes));
    if (!data_) throw std::runtime_error("Failed to allocate data memory");
    std::memcpy(data_, mat.data.get(), data_bytes);
    owns_data_ = true;

    // Load graph
    std::ifstream in(index_path, std::ios::binary);
    if (!in.is_open())
        throw std::runtime_error("Cannot open HNSW index file: " + index_path);

    uint32_t file_npts, file_dim;
    in.read(reinterpret_cast<char*>(&file_npts),          4);
    in.read(reinterpret_cast<char*>(&file_dim),           4);
    in.read(reinterpret_cast<char*>(&M_),                 4);
    in.read(reinterpret_cast<char*>(&M0_),                4);
    in.read(reinterpret_cast<char*>(&efConstruction_),    4);
    in.read(reinterpret_cast<char*>(&max_layer_),         4);
    in.read(reinterpret_cast<char*>(&entry_point_),       4);
    in.read(reinterpret_cast<char*>(&mL_),                sizeof(double));

    if (file_npts != npts_ || file_dim != dim_)
        throw std::runtime_error("Index/data mismatch");

    // Node levels
    node_level_.resize(npts_);
    in.read(reinterpret_cast<char*>(node_level_.data()), npts_ * sizeof(uint32_t));

    // Graph
    const uint32_t MAX_LAYERS = 17;
    graph_.resize(MAX_LAYERS);
    for (uint32_t l = 0; l < MAX_LAYERS; l++)
        graph_[l].resize(npts_);

    locks_ = std::vector<std::mutex>(npts_);

    for (uint32_t l = 0; l <= max_layer_; l++) {
        for (uint32_t i = 0; i < npts_; i++) {
            uint32_t deg;
            in.read(reinterpret_cast<char*>(&deg), 4);
            graph_[l][i].resize(deg);
            if (deg > 0)
                in.read(reinterpret_cast<char*>(graph_[l][i].data()),
                        deg * sizeof(uint32_t));
        }
    }

    std::cout << "HNSW index loaded: " << npts_ << " points, " << dim_
              << " dims, M=" << M_ << ", max_layer=" << max_layer_
              << ", entry_point=" << entry_point_ << std::endl;
}
