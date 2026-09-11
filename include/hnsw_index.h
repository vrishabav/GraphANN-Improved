#pragma once

#include <cstdint>
#include <random>
#include <vector>
#include <string>
#include <mutex>

// Result of a single HNSW query search.
struct HNSWSearchResult {
    std::vector<uint32_t> ids;  // nearest neighbor IDs (sorted by distance, closest first)
    uint32_t dist_cmps;         // number of distance computations
    double latency_us;          // search latency in microseconds
};

// HNSW — Hierarchical Navigable Small World graph index.
//
// Reference: Malkov & Yashunin, "Efficient and robust approximate nearest
// neighbor search using Hierarchical Navigable Small World graphs",
// IEEE TPAMI 2020 (arXiv:1603.09320).
//
// Key concepts:
//   - Multi-layer graph: layer 0 contains ALL nodes; higher layers contain
//     exponentially fewer nodes (each node assigned a random max layer).
//   - Layer assignment: l = floor(-ln(uniform(0,1)) * mL), mL = 1/ln(M)
//   - Construction: for each new point, search top layers greedily (ef=1),
//     then beam-search at the insertion layer and below (ef=efConstruction),
//     selecting M neighbors per layer using heuristic neighbor selection.
//   - Layer 0 uses M0 = 2*M connections (denser base layer).
//   - Search: greedy descent from top layer to layer 1, then beam search
//     at layer 0 with ef=efSearch.
//   - Heuristic neighbor selection (Algorithm 4): keeps candidates that are
//     closer to the new node than to any already-selected neighbor — ensures
//     diversity and long-range navigability.
class HNSWIndex {
public:
    HNSWIndex() = default;
    ~HNSWIndex();

    // ---- Build ----
    // Loads data from an fbin file and builds the HNSW graph.
    //   M:              max connections per layer (layer 0 uses 2*M)
    //   efConstruction: beam width during construction (>= M)
    //   seed:           random seed for layer assignment
    void build(const std::string& data_path, uint32_t M = 16,
               uint32_t efConstruction = 200, uint32_t seed = 42);

    // ---- Search ----
    // Search for K nearest neighbors of a query vector.
    //   query: pointer to query vector (must have dim_ floats)
    //   K:     number of nearest neighbors to return
    //   ef:    search beam width (ef >= K; larger = higher recall, slower)
    HNSWSearchResult search(const float* query, uint32_t K, uint32_t ef) const;

    // ---- Persistence ----
    void save(const std::string& path) const;
    void load(const std::string& index_path, const std::string& data_path);

    uint32_t get_npts()       const { return npts_; }
    uint32_t get_dim()        const { return dim_; }
    uint32_t get_max_layer()  const { return max_layer_; }
    uint32_t get_M()          const { return M_; }
    uint32_t get_efC()        const { return efConstruction_; }

    // Return average degree at each layer (for analysis)
    std::vector<double> layer_avg_degrees() const;

private:
    // A candidate = (distance, node_id). Ordered by distance ascending.
    using Candidate = std::pair<float, uint32_t>;

    // ---- Core algorithms ----

    // Beam search at a single layer starting from entry_point.
    // Returns up to ef candidates sorted by distance (ascending).
    // Used both during construction and search.
    std::vector<Candidate>
    search_layer(const float* query, uint32_t entry_point,
                 uint32_t ef, uint32_t layer) const;

    // Heuristic neighbor selection (Algorithm 4 from the paper).
    // From candidates, select up to M diverse neighbors for node_id at layer.
    // Diversity criterion: candidate c is kept only if
    //   dist(node, c) < dist(c, any_already_selected_neighbor)
    // This ensures the selected set spans different directions.
    std::vector<uint32_t>
    select_neighbors_heuristic(uint32_t node_id,
                                std::vector<Candidate>& candidates,
                                uint32_t M, uint32_t layer,
                                bool extend_candidates = false,
                                bool keep_pruned = true) const;

    // ---- Data ----
    float*   data_    = nullptr;  // contiguous row-major [npts x dim], aligned
    uint32_t npts_    = 0;
    uint32_t dim_     = 0;
    bool     owns_data_ = false;

    // ---- Graph ----
    // graph_[layer][node] = list of neighbor IDs at that layer.
    // Only nodes with node_level_[node] >= layer have entries in layer > 0.
    // Layer 0 is always fully populated.
    std::vector<std::vector<std::vector<uint32_t>>> graph_;  // [layer][node]

    // Per-node maximum layer (inclusive). Assigned at insertion time.
    std::vector<uint32_t> node_level_;

    // Global entry point (node with highest layer assignment).
    uint32_t entry_point_ = 0;
    uint32_t max_layer_   = 0;

    // ---- Parameters ----
    uint32_t M_              = 16;   // max connections per layer (layer 0: 2*M)
    uint32_t M0_             = 32;   // max connections at layer 0 = 2*M
    uint32_t efConstruction_ = 200;  // build beam width
    double   mL_             = 0.0;  // 1/ln(M), controls layer assignment

    // ---- Concurrency ----
    mutable std::vector<std::mutex> locks_;  // per-node locks for parallel build
    mutable std::mutex ep_lock_;            // lock for entry_point_ and max_layer_

    // ---- Helpers ----
    const float* get_vector(uint32_t id) const {
        return data_ + (size_t)id * dim_;
    }

    // Assign a random layer to a new node using the exponential distribution.
    uint32_t random_level(std::mt19937& rng) const;
};
