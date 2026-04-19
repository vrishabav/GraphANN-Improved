#pragma once

#include <cstdint>
#include <vector>
#include <mutex>
#include <string>

// Result of a single query search.
struct SearchResult
{
  std::vector<uint32_t> ids; // nearest neighbor IDs (sorted by distance)
  uint32_t dist_cmps;        // number of distance computations
  double latency_us;         // search latency in microseconds
};

// Statistics about a built graph — used for analysis and reporting.
struct GraphStats {
    double avg_degree;
    uint32_t min_degree;
    uint32_t max_degree;
    double degree_stddev;
    std::vector<uint32_t> degree_hist;  // degree_hist[d] = count of nodes with out-degree d
    double avg_greedy_path_length;
};

// Vamana graph-based approximate nearest neighbor index.
//
// Key concepts:
//   - The graph is built incrementally: each point is inserted by searching
//     the current graph, pruning candidates with the alpha-RNG rule, and
//     adding forward + backward edges.
//   - Greedy search starts from a fixed start node and follows edges to
//     find nearest neighbors, maintaining a candidate list of size L.
//   - The alpha parameter controls edge diversity (alpha > 1 favors long-range
//     edges for better navigability).
//   - R is the max out-degree; gamma*R is the threshold that triggers pruning
//     on neighbor nodes when backward edges are added.
class VamanaIndex
{
public:
  VamanaIndex() = default;
  ~VamanaIndex();

  // ---- Build ----
  // Loads data from an fbin file and builds the Vamana graph.
  //   R:     max out-degree per node
  //   L:     search list size during construction (L >= R)
  //   alpha: RNG pruning parameter (typically 1.0 - 1.5)
  //   gamma: max-degree multiplier for triggering neighbor pruning (e.g. 1.5)
  void build(const std::string &data_path, uint32_t R, uint32_t L,
             float alpha, float gamma, uint32_t num_entry_points = 1,
             bool two_pass = true);

  // ---- Search ----
  // Search for K nearest neighbors of a query vector.
  //   query: pointer to query vector (must have dim_ floats)
  //   K:     number of nearest neighbors to return
  //   L:     search list size (L >= K)
  //   use_quantized: if true, use ADC (asymmetric distance) for traversal
  //                  and re-rank final candidates with exact float32
  SearchResult search(const float *query, uint32_t K, uint32_t L,
                      bool use_quantized = false, bool dynamic_L = false,
                      float dyn_floor_ratio = 0.5f, float dyn_exp_mult = 2.0f,
                      uint32_t dyn_hops = 10, bool use_pca = false) const;

  // ---- Persistence ----
  // Save index (graph + metadata) to a binary file.
  void save(const std::string &path) const;

  // Load index from a binary file. Data file must also be loaded separately.
  void load(const std::string &index_path, const std::string &data_path);

  uint32_t get_npts() const { return npts_; }
  uint32_t get_dim() const { return dim_; }
  uint32_t get_start_node() const { return start_node_; }
  bool has_quantized() const { return has_quantized_; }

  // ---- Analysis ----
  GraphStats compute_graph_stats() const;
  void export_degree_histogram(const std::string &path) const;

  // Build 8-bit scalar quantized representation of the dataset.
  // Must be called after data is loaded (via build() or load()).
  void build_quantized_data();

  // Refine graph edges using quantized distances.
  // Re-prunes each node's neighbor list so surviving edges are robust
  // to quantization noise. Call after build_quantized_data().
  void refine_with_quantization();

  // Build PCA projection: computes top pca_dim principal components and
  // projects all dataset vectors. Used for fast traversal distance computation.
  void build_pca(uint32_t pca_dim = 32);

  // Project a single query vector into PCA space.
  // Result written to out (must have pca_dim_ floats allocated).
  void project_query_pca(const float *query, float *out) const;

private:
  // A candidate = (distance, node_id). Ordered by distance.
  using Candidate = std::pair<float, uint32_t>;

  // ---- Core algorithms ----

  // Greedy search starting from start_node_.
  // Returns (sorted candidate list, visited set V, number of distance computations).
  // visited_out contains ALL nodes visited during traversal (superset of candidates).
  // Pass return_visited=true during index build so robust_prune gets the full V.
  struct GreedyResult
  {
    std::vector<Candidate> candidates; // top-L sorted by distance
    std::vector<Candidate> visited;    // ALL visited nodes (for robust_prune during build)
    uint32_t dist_cmps;
  };
  GreedyResult
  greedy_search(const float *query, uint32_t L, bool return_visited = false,
                bool dynamic_L = false, float dyn_floor_ratio = 0.5f,
                float dyn_exp_mult = 2.0f, uint32_t dyn_hops = 10) const;

  // Greedy search using quantized asymmetric distance.
  // After traversal, re-ranks only the top-K candidates with exact float32 distance.
  GreedyResult
  greedy_search_quantized(const float *query, uint32_t L, uint32_t K, bool dynamic_L = false,
                          float dyn_floor_ratio = 0.5f, float dyn_exp_mult = 2.0f, uint32_t dyn_hops = 10) const;

  // Greedy search using PCA-projected distances for fast traversal.
  // pca_query: pre-projected query (pca_dim_ floats).
  // Caller re-ranks returned candidates with exact float32 distances.
  GreedyResult
  greedy_search_pca(const float *query, const float *pca_query, uint32_t L) const;

  // Alpha-RNG pruning: selects a diverse subset of candidates as neighbors.
  // Modifies graph_[node] in place. Candidates should NOT include node itself.
  void robust_prune(uint32_t node, std::vector<Candidate> &candidates,
                    float alpha, uint32_t R);

  // ---- Data ----
  float *data_ = nullptr; // contiguous row-major [npts x dim], aligned
  uint32_t npts_ = 0;
  uint32_t dim_ = 0;
  bool owns_data_ = false; // whether we allocated data_

  // ---- Quantized data (for ADC search) ----
  uint8_t *quantized_data_ = nullptr; // [npts x dim], row-major uint8
  float *quant_min_ = nullptr;        // per-dimension min [dim]
  float *quant_scale_ = nullptr;      // per-dimension scale [dim]
  bool has_quantized_ = false;

  // ---- PCA projection (for fast approximate traversal) ----
  // Projects 128-dim vectors to pca_dim_-dim space using top eigenvectors.
  // Traversal uses PCA distances; final top-K re-ranked with exact float32.
  float *pca_components_ = nullptr; // [pca_dim x dim] projection matrix
  float *pca_data_ = nullptr;       // [npts x pca_dim] projected dataset
  float *pca_mean_ = nullptr;       // [dim] dataset mean (for centering)
  uint32_t pca_dim_ = 0;
  bool has_pca_ = false;

  // ---- Graph ----
  std::vector<std::vector<uint32_t>> graph_; // adjacency lists
  uint32_t start_node_ = 0;

  // ---- Multiple entry points ----
  // Instead of always starting search from a single medoid, we maintain
  // k cluster medoids as entry points. At query time we pick the closest
  // one (cheap: k distance computations), cutting 3-5 hops off the search.
  std::vector<uint32_t> entry_points_; // medoid of each cluster
  uint32_t num_entry_points_ = 1;      // k; 1 = single medoid (default)

  // Build entry points via k-means on the dataset.
  void build_entry_points(uint32_t k);

  // Pick the best entry point for a given query vector.
  uint32_t best_entry_point(const float *query) const;

  // Reorder node IDs for cache locality: hot nodes (within C BFS hops of
  // medoid) are placed at low indices so they stay in L2/L3 cache.
  void reorder_for_cache(uint32_t C = 3);

  // ---- Concurrency ----
  // Per-node locks for parallel build (mutable so search can be const).
  mutable std::vector<std::mutex> locks_;

  // ---- Helpers ----
  const float *get_vector(uint32_t id) const
  {
    return data_ + (size_t)id * dim_;
  }
  const uint8_t *get_quantized_vector(uint32_t id) const
  {
    return quantized_data_ + (size_t)id * dim_;
  }
};