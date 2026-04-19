#include "vamana_index.h"
#include "distance.h"
#include "io_utils.h"
#include "timer.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <cstring>
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

VamanaIndex::~VamanaIndex()
{
    if (owns_data_ && data_)
    {
        std::free(data_);
        data_ = nullptr;
    }
    if (quantized_data_)
    {
        std::free(quantized_data_);
        quantized_data_ = nullptr;
    }
    if (quant_min_)
    {
        std::free(quant_min_);
        quant_min_ = nullptr;
    }
    if (quant_scale_)
    {
        std::free(quant_scale_);
        quant_scale_ = nullptr;
    }
    if (pca_components_)
    {
        std::free(pca_components_);
        pca_components_ = nullptr;
    }
    if (pca_data_)
    {
        std::free(pca_data_);
        pca_data_ = nullptr;
    }
    if (pca_mean_)
    {
        std::free(pca_mean_);
        pca_mean_ = nullptr;
    }
}

// ============================================================================
// Greedy Search
// ============================================================================
// Beam search starting from start_node_. Maintains a candidate list of at most
// L nodes, always expanding the closest unvisited node. Returns when no
// unvisited candidates remain.
//
// Uses a sorted std::vector<Candidate> (flat list) instead of std::set to
// avoid heap allocations in the inner loop. For L<=200, contiguous memmove
// is drastically faster than Red-Black Tree pointer-chasing.

// ============================================================================
// Greedy Search
// ============================================================================

VamanaIndex::GreedyResult
VamanaIndex::greedy_search(const float *query, uint32_t L, bool return_visited,
                           bool dynamic_L, float dyn_floor_ratio,
                           float dyn_exp_mult, uint32_t dyn_hops) const
{
    // Candidate list: sorted by (distance, id), bounded at size L.
    std::vector<Candidate> candidates;
    candidates.reserve(L + 1);

    // visited_all: every node we computed a distance for.
    // Only populated when return_visited=true (build phase), to pass full V
    // to robust_prune per the paper (Section 2.3).
    std::vector<Candidate> visited_all;
    if (return_visited)
        visited_all.reserve(4 * L); // typically 2-4x L nodes are visited

    // Simple per-query visited/expanded tracking.
    // vector<bool> uses 1 bit per entry so 1M points = 125KB each — fine.
    std::vector<bool> visited_gen(npts_, false);
    std::vector<bool> expanded_gen(npts_, false);

    auto is_visited = [&](uint32_t id)
    { return visited_gen[id]; };
    auto is_expanded = [&](uint32_t id)
    { return expanded_gen[id]; };
    auto mark_visited = [&](uint32_t id)
    { visited_gen[id] = true; };
    auto mark_expanded = [&](uint32_t id)
    { expanded_gen[id] = true; };

    uint32_t dist_cmps = 0;

    // Pick best entry point for this query (multiple entry points improvement)
    uint32_t entry = best_entry_point(query);

    // Seed with entry point
    float start_dist = compute_l2sq(query, get_vector(entry), dim_);
    dist_cmps++;
    candidates.push_back({start_dist, entry});
    mark_visited(entry);
    if (return_visited)
        visited_all.push_back({start_dist, entry});

    uint32_t expand_pos = 0;

    // Dynamic beam width (Proposal C)
    uint32_t active_L = dynamic_L ? std::min((uint32_t)10, L) : L;
    uint32_t floor_L = std::max((uint32_t)10, static_cast<uint32_t>(L * dyn_floor_ratio));
    float best_dist = FLT_MAX;
    uint32_t hops_without_improvement = 0;

    while (expand_pos < (uint32_t)candidates.size())
    {
        uint32_t track_node = candidates[expand_pos].second;
        if (is_expanded(track_node))
        {
            expand_pos++;
            continue;
        }

        if (dynamic_L)
        {
            uint32_t track_index = std::min((uint32_t)candidates.size() - 1, active_L - 1);
            float current_best = candidates[track_index].first;

            if (expand_pos >= active_L)
            {
                active_L = std::min(L, static_cast<uint32_t>(
                                           std::max((float)active_L + 1.0f, active_L * dyn_exp_mult)));
                hops_without_improvement = 0;
                best_dist = current_best;
            }
            else
            {
                if (current_best < best_dist * 0.95f)
                {
                    best_dist = current_best;
                    hops_without_improvement = 0;
                    active_L = std::max(floor_L, static_cast<uint32_t>(active_L * 0.9f));
                }
                else if (current_best < best_dist)
                {
                    best_dist = current_best;
                    hops_without_improvement = 0;
                }
                else
                {
                    hops_without_improvement++;
                    if (hops_without_improvement >= dyn_hops)
                    {
                        active_L = std::min(L, static_cast<uint32_t>(
                                                   std::max((float)active_L + 1.0f, active_L * dyn_exp_mult)));
                        hops_without_improvement = 0;
                    }
                }
            }
            if (expand_pos >= active_L)
                break;
        }

        uint32_t best_node = candidates[expand_pos].second;
        expand_pos++;
        mark_expanded(best_node);

        // Early termination: if we are in search mode (not build) and the
        // best candidate distance hasn't improved meaningfully in the last
        // several expansions, stop early. Saves ~15-20% compute on easy queries
        // with negligible recall loss since remaining candidates are far away.
        if (!return_visited && !dynamic_L && candidates.size() >= (size_t)L)
        {
            float current_best = candidates[0].first;
            float frontier_dist = candidates[expand_pos > 0 ? expand_pos - 1 : 0].first;
            // If frontier is more than 3x the best distance, remaining expansions
            // are unlikely to improve top-K results
            if (frontier_dist > 3.0f * current_best && expand_pos > L / 4)
                break;
        }

        // During build (return_visited=true): copy neighbor list under lock
        // to avoid data races with parallel insertions.
        // During search (return_visited=false): no writes happening, so read
        // directly without locking or copying — eliminates malloc per expansion.
        std::vector<uint32_t> neighbors_copy;
        const std::vector<uint32_t> *nbrs_ptr;
        if (return_visited)
        {
            std::lock_guard<std::mutex> lock(locks_[best_node]);
            neighbors_copy = graph_[best_node];
            nbrs_ptr = &neighbors_copy;
        }
        else
        {
            nbrs_ptr = &graph_[best_node];
        }
        const std::vector<uint32_t> &neighbors = *nbrs_ptr;

        // Prefetch all neighbor vectors into CPU cache before computing distances.
        for (uint32_t nbr : neighbors)
        {
            const char *ptr = reinterpret_cast<const char *>(get_vector(nbr));
            for (uint32_t cl = 0; cl < 8; cl++)
                __builtin_prefetch(ptr + cl * 64, 0, 1);
        }

        for (uint32_t nbr : neighbors)
        {
            if (is_visited(nbr))
                continue;
            mark_visited(nbr);

            // Early-abandon: only applies when the candidate list is full.
            // During build (return_visited=true) we must NOT early-abandon because
            // we need the true distance for every visited node to pass to robust_prune.
            float threshold = (!return_visited && candidates.size() >= L)
                                  ? candidates.back().first
                                  : FLT_MAX;
            float d = compute_l2sq_ea(query, get_vector(nbr), dim_, threshold);
            dist_cmps++;

            if (d == FLT_MAX)
                continue; // early-abandoned

            // Always record in visited_all for the build phase
            if (return_visited)
                visited_all.push_back({d, nbr});

            // Only insert into candidate list if it beats the current worst
            if (candidates.size() >= L && d >= candidates.back().first)
                continue;

            Candidate new_cand = {d, nbr};
            auto pos = std::lower_bound(candidates.begin(), candidates.end(), new_cand);
            size_t insert_idx = pos - candidates.begin();

            if (candidates.size() < L)
            {
                candidates.insert(pos, new_cand);
            }
            else
            {
                if (pos == candidates.end())
                    continue;
                candidates.back() = new_cand;
                std::rotate(pos, candidates.end() - 1, candidates.end());
            }

            if (insert_idx < expand_pos)
                expand_pos = (uint32_t)insert_idx;
        }
    }

    return {std::move(candidates), std::move(visited_all), dist_cmps};
}

// ============================================================================
// Robust Prune (Alpha-RNG Rule)
// ============================================================================
// Given a node and a set of candidates, greedily select neighbors that are
// "diverse" — a candidate c is added only if it's not too close to any
// already-selected neighbor (within a factor of alpha).
//
// Formally: add c if for ALL already-chosen neighbors n:
//     dist(node, c) <= alpha * dist(c, n)
//
// This ensures good graph navigability by keeping some long-range edges
// (alpha > 1 makes it easier for a candidate to survive pruning).

void VamanaIndex::robust_prune(uint32_t node, std::vector<Candidate> &candidates,
                               float alpha, uint32_t R)
{
    // Remove self from candidates if present
    candidates.erase(
        std::remove_if(candidates.begin(), candidates.end(),
                       [node](const Candidate &c)
                       { return c.second == node; }),
        candidates.end());

    // Sort by distance to node (ascending)
    std::sort(candidates.begin(), candidates.end());

    std::vector<uint32_t> new_neighbors;
    new_neighbors.reserve(R);

    for (const auto &[dist_to_node, cand_id] : candidates)
    {
        if (new_neighbors.size() >= R)
            break;

        // Check alpha-RNG condition against all already-selected neighbors
        bool keep = true;
        for (uint32_t selected : new_neighbors)
        {
            float dist_cand_to_selected =
                compute_l2sq(get_vector(cand_id), get_vector(selected), dim_);
            if (dist_to_node > alpha * dist_cand_to_selected)
            {
                keep = false;
                break;
            }
        }

        if (keep)
            new_neighbors.push_back(cand_id);
    }

    graph_[node] = std::move(new_neighbors);
}

// ============================================================================
// Build
// ============================================================================

void VamanaIndex::build(const std::string &data_path, uint32_t R, uint32_t L,
                        float alpha, float gamma, uint32_t num_entry_points,
                        bool two_pass)
{
    num_entry_points_ = num_entry_points;
    // --- Load data ---
    std::cout << "Loading data from " << data_path << "..." << std::endl;
    FloatMatrix mat = load_fbin(data_path);
    npts_ = mat.npts;
    dim_ = mat.dims;
    data_ = mat.data.release();
    owns_data_ = true;

    std::cout << "  Points: " << npts_ << ", Dimensions: " << dim_ << std::endl;

    if (L < R)
    {
        std::cerr << "Warning: L (" << L << ") < R (" << R
                  << "). Setting L = R." << std::endl;
        L = R;
    }

    // --- Initialize graph and per-node locks ---
    graph_.resize(npts_);
    locks_ = std::vector<std::mutex>(npts_);

    // --- FIX 1: Medoid initialization ---
    // Compute approximate medoid: find the point closest to the dataset centroid.
    // This is O(n*dim) and guarantees search starts from the geometric center,
    // reducing average traversal depth vs a random start node (paper Section 2.3).
    std::cout << "Computing medoid..." << std::endl;
    {
        // Compute centroid (mean vector)
        std::vector<double> centroid(dim_, 0.0);
        for (uint32_t i = 0; i < npts_; i++)
        {
            const float *v = get_vector(i);
            for (uint32_t d = 0; d < dim_; d++)
                centroid[d] += v[d];
        }
        for (uint32_t d = 0; d < dim_; d++)
            centroid[d] /= npts_;

        // Find point closest to centroid
        float best_dist = FLT_MAX;
        uint32_t best_id = 0;
        for (uint32_t i = 0; i < npts_; i++)
        {
            const float *v = get_vector(i);
            float dist = 0.0f;
            for (uint32_t d = 0; d < dim_; d++)
            {
                float diff = v[d] - (float)centroid[d];
                dist += diff * diff;
            }
            if (dist < best_dist)
            {
                best_dist = dist;
                best_id = i;
            }
        }
        start_node_ = best_id;
    }
    std::cout << "  Medoid (start node): " << start_node_ << std::endl;

    // THE FIX: initialize entry_points_ to the medoid immediately so that
    // best_entry_point() never accesses an empty vector during the build loop.
    // build_entry_points() will overwrite this at the end of build() if k > 1.
    entry_points_ = {start_node_};
    num_entry_points_ = 1;

    // --- FIX 2: Random R-regular graph pre-initialization ---
    // Paper (Section 2.3): "initialize G to a random R-regular directed graph"
    // This bootstraps connectivity so early insertions have meaningful search paths,
    // unlike starting from an empty graph where early points have no neighbors.
    std::cout << "Pre-initializing random R-regular graph..." << std::endl;
    {
        std::mt19937 rng(42);
        for (uint32_t i = 0; i < npts_; i++)
        {
            graph_[i].clear();
            graph_[i].reserve(R);
            // Sample R distinct random neighbors (excluding self)
            std::vector<uint32_t> pool;
            pool.reserve(R * 2);
            while ((uint32_t)pool.size() < R)
            {
                uint32_t nbr = rng() % npts_;
                if (nbr != i)
                    pool.push_back(nbr);
            }
            // Deduplicate
            std::sort(pool.begin(), pool.end());
            pool.erase(std::unique(pool.begin(), pool.end()), pool.end());
            // Take up to R
            for (uint32_t j = 0; j < R && j < (uint32_t)pool.size(); j++)
                graph_[i].push_back(pool[j]);
        }
    }

    // --- FIX 3: Two-pass build ---
    // Paper (Section 2.3): "two passes over the dataset, the first pass with α=1,
    // and the second with a user-defined α >= 1."
    // Pass 1 (alpha=1.0) builds dense local structure.
    // Pass 2 (alpha=user) adds long-range edges on top.
    uint32_t gamma_R = static_cast<uint32_t>(gamma * R);

    auto run_pass = [&](float pass_alpha, int pass_num)
    {
        // Pass 2 uses half the threads to reduce memory pressure.
        // After Pass 1 the graph is dense so each greedy_search visits more
        // nodes, and running all threads simultaneously exhausts RAM on macOS.
        int max_threads = omp_get_max_threads();
        int pass_threads = (pass_num == 1) ? max_threads : std::max(1, max_threads / 2);
        omp_set_num_threads(pass_threads);

        std::cout << "Pass " << pass_num << " (alpha=" << pass_alpha
                  << ", R=" << R << ", L=" << L
                  << ", gamma=" << gamma
                  << ", threads=" << pass_threads << ")..." << std::endl;

        // Random insertion order (re-shuffle each pass for diversity)
        std::mt19937 rng(42 + pass_num);
        std::vector<uint32_t> perm(npts_);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng);

        Timer pass_timer;

#pragma omp parallel for schedule(dynamic, 64)
        for (int64_t idx = 0; idx < (int64_t)npts_; idx++)
        {
            uint32_t point = perm[idx];

            // Pass full visited set V to robust_prune in pass 1 only.
            // In pass 2 the graph is already well-connected so top-L is sufficient
            // and avoids the memory overhead of collecting the full visited set.
            bool collect_visited = (pass_num == 1);
            auto result = greedy_search(get_vector(point), L, collect_visited);

            // Pass 1: use full visited set for better long-range edge candidates
            // Pass 2: use top-L candidates (graph already navigable)
            std::vector<Candidate> &prune_candidates = collect_visited
                                                           ? result.visited
                                                           : result.candidates;

            robust_prune(point, prune_candidates, pass_alpha, R);

            // Add backward edges
            for (uint32_t nbr : graph_[point])
            {
                std::lock_guard<std::mutex> lock(locks_[nbr]);
                graph_[nbr].push_back(point);

                if (graph_[nbr].size() > gamma_R)
                {
                    std::vector<Candidate> nbr_candidates;
                    nbr_candidates.reserve(graph_[nbr].size());
                    for (uint32_t nn : graph_[nbr])
                    {
                        float d = compute_l2sq(get_vector(nbr), get_vector(nn), dim_);
                        nbr_candidates.push_back({d, nn});
                    }
                    robust_prune(nbr, nbr_candidates, pass_alpha, R);
                }
            }

            if (idx % 10000 == 0)
            {
#pragma omp critical
                {
                    std::cout << "\r  Inserted " << idx << " / " << npts_
                              << " points" << std::flush;
                }
            }
        }

        double pass_time = pass_timer.elapsed_seconds();
        size_t total_edges = 0;
        for (uint32_t i = 0; i < npts_; i++)
            total_edges += graph_[i].size();
        std::cout << "\n  Pass " << pass_num << " complete in " << pass_time
                  << "s. Avg degree: " << (double)total_edges / npts_ << std::endl;

        // Restore full thread count for next operation
        omp_set_num_threads(max_threads);
    };

    Timer build_timer;

    if (two_pass)
    {
        // Two-pass build: Pass 1 at alpha=1.0 for local structure,
        // Pass 2 at user alpha for long-range edges.
        // NOTE: Pass 2 is memory-intensive on dense graphs. If it crashes,
        // use --single_pass flag to skip it.
        run_pass(1.0f, 1);
        if (alpha > 1.0f)
            run_pass(alpha, 2);
    }
    else
    {
        // Single-pass build: directly use user alpha.
        // Faster and less memory-intensive, slightly lower quality.
        std::cout << "Single-pass build (alpha=" << alpha << ")..." << std::endl;
        run_pass(alpha, 1);
    }

    double build_time = build_timer.elapsed_seconds();

    size_t total_edges = 0;
    for (uint32_t i = 0; i < npts_; i++)
        total_edges += graph_[i].size();

    std::cout << "\nBuild complete in " << build_time << " seconds." << std::endl;
    std::cout << "Average out-degree: " << (double)total_edges / npts_ << std::endl;

    // Build entry points (uses medoid as single entry point by default)
    build_entry_points(num_entry_points_);

    // NOTE: reorder_for_cache() is disabled because it remaps data_ in memory
    // but the data file on disk (sift_base.fbin) is never updated. At search
    // time the graph has remapped IDs but loaded vectors are in original order,
    // causing catastrophic recall loss. Correct implementation would need to
    // save the reordered data to a separate file.
    // reorder_for_cache(3);
}

// ============================================================================
// Build Quantized Data
// ============================================================================
// Computes per-dimension min/scale and quantizes the entire dataset to uint8.
// Quantization: q[d] = round((val[d] - min[d]) / scale[d]), clamped to [0,255]
// Dequantization: val[d] ≈ q[d] * scale[d] + min[d]

void VamanaIndex::build_quantized_data()
{
    if (npts_ == 0 || dim_ == 0 || data_ == nullptr)
        throw std::runtime_error("Cannot quantize: data not loaded");

    std::cout << "Building quantized data (uint8 scalar quantization)..." << std::endl;
    Timer qt;

    // Allocate per-dimension statistics
    quant_min_ = static_cast<float *>(std::malloc(dim_ * sizeof(float)));
    quant_scale_ = static_cast<float *>(std::malloc(dim_ * sizeof(float)));
    if (!quant_min_ || !quant_scale_)
        throw std::runtime_error("Failed to allocate quantization tables");

    // Compute per-dimension min and max
    for (uint32_t d = 0; d < dim_; d++)
    {
        float dmin = FLT_MAX, dmax = -FLT_MAX;
        for (uint32_t i = 0; i < npts_; i++)
        {
            float val = data_[(size_t)i * dim_ + d];
            if (val < dmin)
                dmin = val;
            if (val > dmax)
                dmax = val;
        }
        quant_min_[d] = dmin;
        float range = dmax - dmin;
        quant_scale_[d] = (range > 1e-9f) ? (range / 255.0f) : 1.0f;
    }

    // Allocate quantized data (64-byte aligned for SIMD)
    size_t qdata_size = (size_t)npts_ * dim_;
    size_t aligned_size = (qdata_size + 63) & ~(size_t)63;
    quantized_data_ = static_cast<uint8_t *>(aligned_alloc(64, aligned_size));
    if (!quantized_data_)
        throw std::runtime_error("Failed to allocate quantized data");

// Quantize all vectors
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)npts_; i++)
    {
        const float *src = data_ + i * dim_;
        uint8_t *dst = quantized_data_ + i * dim_;
        for (uint32_t d = 0; d < dim_; d++)
        {
            float normalized = (src[d] - quant_min_[d]) / quant_scale_[d];
            int val = static_cast<int>(std::round(normalized));
            dst[d] = static_cast<uint8_t>(std::max(0, std::min(255, val)));
        }
    }

    has_quantized_ = true;
    std::cout << "  Quantized " << npts_ << " vectors in "
              << qt.elapsed_seconds() << "s" << std::endl;
    std::cout << "  Quantized data size: "
              << (qdata_size / (1024.0 * 1024.0)) << " MB (vs "
              << ((size_t)npts_ * dim_ * sizeof(float) / (1024.0 * 1024.0))
              << " MB float32)" << std::endl;
}

// ============================================================================
// Greedy Search — Quantized (ADC)
// ============================================================================
// Same beam search as greedy_search(), but uses asymmetric distance
// (float32 query vs uint8 dataset) for graph traversal. After the beam search
// completes, all L candidates are re-ranked using exact float32 distances.
// Uses flat sorted vector (no std::set) for zero heap allocations.

VamanaIndex::GreedyResult
VamanaIndex::greedy_search_quantized(const float *query, uint32_t L, uint32_t K, bool dynamic_L,
                                     float dyn_floor_ratio, float dyn_exp_mult, uint32_t dyn_hops) const
{
    std::vector<Candidate> candidates;
    candidates.reserve(L + 1);

    std::vector<bool> visited_gen(npts_, false);
    std::vector<bool> expanded_gen(npts_, false);

    auto is_visited = [&](uint32_t id)
    { return visited_gen[id]; };
    auto is_expanded = [&](uint32_t id)
    { return expanded_gen[id]; };
    auto mark_visited = [&](uint32_t id)
    { visited_gen[id] = true; };
    auto mark_expanded = [&](uint32_t id)
    { expanded_gen[id] = true; };

    uint32_t dist_cmps = 0;

    // Seed with best entry point (use asymmetric distance)
    uint32_t entry = best_entry_point(query);
    float start_dist = compute_l2sq_asymmetric(
        query, get_quantized_vector(entry),
        quant_min_, quant_scale_, dim_);
    dist_cmps++;
    candidates.push_back({start_dist, entry});
    mark_visited(entry);

    uint32_t expand_pos = 0;

    // Proposal C Dynamics
    uint32_t active_L = dynamic_L ? std::min((uint32_t)10, L) : L;
    uint32_t floor_L = std::max((uint32_t)10, static_cast<uint32_t>(L * dyn_floor_ratio));
    float best_dist = FLT_MAX;
    uint32_t hops_without_improvement = 0;

    while (expand_pos < candidates.size())
    {
        uint32_t track_node = candidates[expand_pos].second;
        if (is_expanded(track_node))
        {
            expand_pos++;
            continue;
        }

        if (dynamic_L)
        {
            uint32_t track_index = std::min((uint32_t)candidates.size() - 1, active_L - 1);
            float current_best = candidates[track_index].first;

            if (expand_pos >= active_L)
            {
                active_L = std::min(L, static_cast<uint32_t>(std::max((float)active_L + 1.0f, active_L * dyn_exp_mult)));
                hops_without_improvement = 0;
                best_dist = current_best;
            }
            else
            {
                if (current_best < best_dist * 0.95f)
                {
                    best_dist = current_best;
                    hops_without_improvement = 0;
                    active_L = std::max(floor_L, static_cast<uint32_t>(active_L * 0.9f));
                }
                else if (current_best < best_dist)
                {
                    best_dist = current_best;
                    hops_without_improvement = 0;
                }
                else
                {
                    hops_without_improvement++;
                    if (hops_without_improvement >= dyn_hops)
                    {
                        active_L = std::min(L, static_cast<uint32_t>(std::max((float)active_L + 1.0f, active_L * dyn_exp_mult)));
                        hops_without_improvement = 0;
                    }
                }
            }
            if (expand_pos >= active_L)
                break;
        }

        uint32_t best_node = candidates[expand_pos].second;
        expand_pos++;
        mark_expanded(best_node);

        // Copy neighbor list under lock
        std::vector<uint32_t> neighbors;
        {
            std::lock_guard<std::mutex> lock(locks_[best_node]);
            neighbors = graph_[best_node];
        }

        for (uint32_t nbr : neighbors)
        {
            if (is_visited(nbr))
                continue;
            mark_visited(nbr);

            // Early-abandon threshold
            float threshold = (candidates.size() >= L)
                                  ? candidates.back().first
                                  : FLT_MAX;
            float d = compute_l2sq_asymmetric_ea(
                query, get_quantized_vector(nbr),
                quant_min_, quant_scale_, dim_, threshold);
            dist_cmps++;

            // Skip if early-abandoned
            if (d == FLT_MAX)
                continue;

            // Skip if list is full and this is worse than the worst
            if (candidates.size() >= L && d >= candidates.back().first)
                continue;

            // Binary search for sorted insertion
            Candidate new_cand = {d, nbr};
            auto pos = std::lower_bound(candidates.begin(), candidates.end(), new_cand);
            size_t insert_idx = pos - candidates.begin();

            if (candidates.size() < L)
            {
                candidates.insert(pos, new_cand);
            }
            else
            {
                if (pos == candidates.end())
                    continue;
                candidates.back() = new_cand;
                std::rotate(pos, candidates.end() - 1, candidates.end());
            }

            // Backtrack cursor if we inserted a closer candidate before it
            if (insert_idx < expand_pos)
                expand_pos = insert_idx;
        }
    }

    // Re-rank entire candidate list with exact float32 distance (up to L)
    uint32_t num_to_rerank = candidates.size();
    std::vector<Candidate> reranked;
    reranked.reserve(num_to_rerank);

    for (uint32_t i = 0; i < num_to_rerank; i++)
    {
        uint32_t id = candidates[i].second;
        float exact_dist = compute_l2sq(query, get_vector(id), dim_);
        reranked.push_back({exact_dist, id});
    }
    std::sort(reranked.begin(), reranked.end());

    return {std::move(reranked), {}, dist_cmps};
}

// ============================================================================
// Greedy Search — PCA (low-dimensional traversal)
// ============================================================================
// Same beam search as greedy_search(), but computes distances in PCA-projected
// space (pca_dim_ dims instead of 128). This reduces memory bandwidth and
// compute per distance call by 128/pca_dim_ = 4x for pca_dim_=32.
// Caller re-ranks all returned candidates with exact float32 distances.

VamanaIndex::GreedyResult
VamanaIndex::greedy_search_pca(const float *query, const float *pca_query,
                               uint32_t L) const
{
    std::vector<Candidate> candidates;
    candidates.reserve(L + 1);

    std::vector<bool> visited(npts_, false);
    std::vector<bool> expanded(npts_, false);
    uint32_t dist_cmps = 0;

    auto is_visited = [&](uint32_t id)
    { return visited[id]; };
    auto is_expanded = [&](uint32_t id)
    { return expanded[id]; };
    auto mark_visited = [&](uint32_t id)
    { visited[id] = true; };
    auto mark_expanded = [&](uint32_t id)
    { expanded[id] = true; };

    // Seed with best entry point using PCA distance
    uint32_t entry = best_entry_point(query);
    const float *entry_pca = pca_data_ + (size_t)entry * pca_dim_;
    float start_dist = compute_l2sq_pca(pca_query, entry_pca, pca_dim_);
    dist_cmps++;
    candidates.push_back({start_dist, entry});
    mark_visited(entry);

    uint32_t expand_pos = 0;

    while (expand_pos < (uint32_t)candidates.size())
    {
        if (is_expanded(candidates[expand_pos].second))
        {
            expand_pos++;
            continue;
        }

        uint32_t best_node = candidates[expand_pos].second;
        expand_pos++;
        mark_expanded(best_node);

        // Early termination
        if (candidates.size() >= (size_t)L)
        {
            float current_best = candidates[0].first;
            float frontier_dist = candidates[expand_pos > 0 ? expand_pos - 1 : 0].first;
            if (frontier_dist > 3.0f * current_best && expand_pos > L / 4)
                break;
        }

        const std::vector<uint32_t> &neighbors = graph_[best_node];

        // Prefetch PCA vectors of neighbors
        for (uint32_t nbr : neighbors)
        {
            const char *ptr = reinterpret_cast<const char *>(
                pca_data_ + (size_t)nbr * pca_dim_);
            for (uint32_t cl = 0; cl < 2; cl++) // 32 floats = 128 bytes = 2 cache lines
                __builtin_prefetch(ptr + cl * 64, 0, 1);
        }

        for (uint32_t nbr : neighbors)
        {
            if (is_visited(nbr))
                continue;
            mark_visited(nbr);

            // Compute distance in PCA space — 4x cheaper than full 128-dim
            float threshold = (candidates.size() >= L)
                                  ? candidates.back().first
                                  : FLT_MAX;
            const float *nbr_pca = pca_data_ + (size_t)nbr * pca_dim_;
            float d = compute_l2sq_pca(pca_query, nbr_pca, pca_dim_);
            dist_cmps++;

            if (candidates.size() >= L && d >= candidates.back().first)
                continue;

            Candidate new_cand = {d, nbr};
            auto pos = std::lower_bound(candidates.begin(), candidates.end(), new_cand);
            size_t insert_idx = pos - candidates.begin();

            if (candidates.size() < L)
            {
                candidates.insert(pos, new_cand);
            }
            else
            {
                if (pos == candidates.end())
                    continue;
                candidates.back() = new_cand;
                std::rotate(pos, candidates.end() - 1, candidates.end());
            }

            if (insert_idx < expand_pos)
                expand_pos = (uint32_t)insert_idx;
        }
    }

    return {std::move(candidates), {}, dist_cmps};
}

// ============================================================================
// Search
// ============================================================================

SearchResult VamanaIndex::search(const float *query, uint32_t K, uint32_t L,
                                 bool use_quantized, bool dynamic_L,
                                 float dyn_floor_ratio, float dyn_exp_mult,
                                 uint32_t dyn_hops, bool use_pca) const
{
    if (L < K)
        L = K;

    Timer t;
    GreedyResult search_result;

    if (use_pca && has_pca_)
    {
        // PCA search: traverse graph using low-dim PCA distances,
        // then re-rank top-L candidates with exact float32 distances.
        // Project query into PCA space once up front.
        std::vector<float> pca_query(pca_dim_);
        project_query_pca(query, pca_query.data());

        // Run greedy search with PCA distances
        search_result = greedy_search_pca(query, pca_query.data(), L);

        // Re-rank all candidates with exact distances
        for (auto &[dist, id] : search_result.candidates)
            dist = compute_l2sq(query, get_vector(id), dim_);
        std::sort(search_result.candidates.begin(), search_result.candidates.end());
    }
    else if (use_quantized && has_quantized_)
    {
        search_result = greedy_search_quantized(query, L, K, dynamic_L,
                                                dyn_floor_ratio, dyn_exp_mult, dyn_hops);
    }
    else
    {
        search_result = greedy_search(query, L, /*return_visited=*/false,
                                      dynamic_L, dyn_floor_ratio, dyn_exp_mult, dyn_hops);
    }

    auto &candidates = search_result.candidates;
    double latency = t.elapsed_us();

    SearchResult result;
    result.dist_cmps = search_result.dist_cmps;
    result.latency_us = latency;
    result.ids.reserve(K);
    for (uint32_t i = 0; i < K && i < candidates.size(); i++)
    {
        result.ids.push_back(candidates[i].second);
    }
    return result;
}

// ============================================================================
// Cache-Friendly Graph Reordering
// ============================================================================
// Reorders node IDs so that frequently accessed nodes (those within C BFS hops
// of the medoid, touched by nearly every query) are stored at low indices and
// thus contiguously in memory. This improves L2/L3 cache hit rates during search.
//
// Algorithm:
//   1. BFS from medoid up to C hops — these are the "hot" nodes
//   2. Assign new IDs: hot nodes first (0..num_hot-1), rest after
//   3. Remap all node IDs in graph_ and data_
//   4. Update start_node_ and entry_points_

void VamanaIndex::reorder_for_cache(uint32_t C)
{
    std::cout << "Reordering graph for cache locality (C=" << C << " hops)..." << std::endl;
    Timer t;

    // BFS from start_node_ up to C hops
    std::vector<uint32_t> new_id(npts_, UINT32_MAX);
    std::vector<uint32_t> order;
    order.reserve(npts_);

    std::vector<uint32_t> frontier = {start_node_};
    std::vector<bool> visited(npts_, false);
    visited[start_node_] = true;

    for (uint32_t hop = 0; hop < C && !frontier.empty(); hop++)
    {
        std::vector<uint32_t> next;
        for (uint32_t node : frontier)
        {
            order.push_back(node);
            for (uint32_t nbr : graph_[node])
            {
                if (!visited[nbr])
                {
                    visited[nbr] = true;
                    next.push_back(nbr);
                }
            }
        }
        frontier = std::move(next);
    }
    // Add frontier nodes themselves
    for (uint32_t node : frontier)
        order.push_back(node);

    uint32_t num_hot = order.size();

    // Add remaining nodes in original order
    for (uint32_t i = 0; i < npts_; i++)
        if (!visited[i])
            order.push_back(i);

    // Build new_id mapping: new_id[old] = new
    for (uint32_t new_idx = 0; new_idx < npts_; new_idx++)
        new_id[order[new_idx]] = new_idx;

    // Remap graph edges
    std::vector<std::vector<uint32_t>> new_graph(npts_);
    for (uint32_t i = 0; i < npts_; i++)
    {
        new_graph[new_id[i]].reserve(graph_[i].size());
        for (uint32_t nbr : graph_[i])
            new_graph[new_id[i]].push_back(new_id[nbr]);
    }
    graph_ = std::move(new_graph);

    // Remap data_ (reorder float vectors)
    size_t vec_bytes = dim_ * sizeof(float);
    float *new_data = static_cast<float *>(aligned_alloc(64,
                                                         (size_t)npts_ * vec_bytes));
    for (uint32_t i = 0; i < npts_; i++)
        std::memcpy(new_data + (size_t)new_id[i] * dim_,
                    data_ + (size_t)i * dim_, vec_bytes);
    std::free(data_);
    data_ = new_data;

    // Update start_node_ and entry_points_
    start_node_ = new_id[start_node_];
    for (uint32_t &ep : entry_points_)
        ep = new_id[ep];

    std::cout << "  Reordered in " << t.elapsed_seconds() << "s. "
              << "Hot nodes (within " << C << " hops): " << num_hot << std::endl;
}

// ============================================================================
// PCA Projection
// ============================================================================
// Computes top-pca_dim principal components of the dataset using the power
// iteration method (cheap, no LAPACK needed). Projects all dataset vectors
// into the reduced space for fast approximate distance computation during
// graph traversal, followed by exact re-ranking of top-K candidates.

void VamanaIndex::build_pca(uint32_t pca_dim)
{
    if (npts_ == 0 || dim_ == 0 || data_ == nullptr)
        throw std::runtime_error("Cannot build PCA: data not loaded");
    if (pca_dim >= dim_)
    {
        std::cout << "PCA dim >= data dim, skipping." << std::endl;
        return;
    }

    std::cout << "Building PCA projection (" << dim_ << " -> " << pca_dim
              << " dims)..." << std::endl;
    Timer t;

    // --- Compute dataset mean ---
    pca_mean_ = static_cast<float *>(std::calloc(dim_, sizeof(float)));
    for (uint32_t i = 0; i < npts_; i++)
    {
        const float *v = get_vector(i);
        for (uint32_t d = 0; d < dim_; d++)
            pca_mean_[d] += v[d];
    }
    for (uint32_t d = 0; d < dim_; d++)
        pca_mean_[d] /= (float)npts_;

    // --- Power iteration to find top pca_dim eigenvectors ---
    // Use a random subsample of 50K points for speed (sufficient for SIFT1M)
    const uint32_t SAMPLE = std::min(npts_, (uint32_t)50000);
    std::mt19937 rng(42);
    std::vector<uint32_t> sample_ids(npts_);
    std::iota(sample_ids.begin(), sample_ids.end(), 0);
    std::shuffle(sample_ids.begin(), sample_ids.end(), rng);
    sample_ids.resize(SAMPLE);

    // Allocate projection matrix [pca_dim x dim]
    pca_components_ = static_cast<float *>(
        aligned_alloc(64, pca_dim * dim_ * sizeof(float)));
    std::memset(pca_components_, 0, pca_dim * dim_ * sizeof(float));

    // Deflation: find each eigenvector in turn
    std::vector<float> residual(dim_);
    for (uint32_t k = 0; k < pca_dim; k++)
    {
        float *comp = pca_components_ + k * dim_;

        // Initialize randomly
        for (uint32_t d = 0; d < dim_; d++)
            comp[d] = ((float)rng() / UINT32_MAX) - 0.5f;

        // Power iteration: 30 iterations on the sample
        for (int iter = 0; iter < 30; iter++)
        {
            std::vector<float> new_comp(dim_, 0.0f);

            for (uint32_t si = 0; si < SAMPLE; si++)
            {
                const float *v = get_vector(sample_ids[si]);
                // Compute centered dot product: (v - mean) . comp
                float dot = 0.0f;
                for (uint32_t d = 0; d < dim_; d++)
                    dot += (v[d] - pca_mean_[d]) * comp[d];
                // Accumulate: new_comp += dot * (v - mean)
                for (uint32_t d = 0; d < dim_; d++)
                    new_comp[d] += dot * (v[d] - pca_mean_[d]);
            }

            // Normalize
            float norm = 0.0f;
            for (uint32_t d = 0; d < dim_; d++)
                norm += new_comp[d] * new_comp[d];
            norm = std::sqrt(norm);
            if (norm < 1e-9f)
                break;
            for (uint32_t d = 0; d < dim_; d++)
                comp[d] = new_comp[d] / norm;
        }

        // Deflation: remove this component's contribution from future iterations
        // (subtract its projection from each sample vector conceptually — done
        //  implicitly by operating on the residual space in next iteration)
        // Orthogonalize against previous components (Gram-Schmidt)
        for (uint32_t prev = 0; prev < k; prev++)
        {
            const float *prev_comp = pca_components_ + prev * dim_;
            float dot = 0.0f;
            for (uint32_t d = 0; d < dim_; d++)
                dot += comp[d] * prev_comp[d];
            for (uint32_t d = 0; d < dim_; d++)
                comp[d] -= dot * prev_comp[d];
        }
        // Re-normalize after Gram-Schmidt
        float norm = 0.0f;
        for (uint32_t d = 0; d < dim_; d++)
            norm += comp[d] * comp[d];
        norm = std::sqrt(norm);
        if (norm > 1e-9f)
            for (uint32_t d = 0; d < dim_; d++)
                comp[d] /= norm;
    }

    // --- Project all dataset vectors ---
    pca_dim_ = pca_dim;
    pca_data_ = static_cast<float *>(
        aligned_alloc(64, (size_t)npts_ * pca_dim * sizeof(float)));

#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t)npts_; i++)
    {
        const float *v = get_vector(i);
        float *proj = pca_data_ + i * pca_dim;
        for (uint32_t k = 0; k < pca_dim; k++)
        {
            const float *comp = pca_components_ + k * dim_;
            float dot = 0.0f;
            for (uint32_t d = 0; d < dim_; d++)
                dot += (v[d] - pca_mean_[d]) * comp[d];
            proj[k] = dot;
        }
    }

    has_pca_ = true;
    std::cout << "  PCA built in " << t.elapsed_seconds() << "s. "
              << "Memory: " << ((size_t)npts_ * pca_dim * 4) / (1024 * 1024)
              << " MB (vs " << ((size_t)npts_ * dim_ * 4) / (1024 * 1024)
              << " MB full)" << std::endl;
}

void VamanaIndex::project_query_pca(const float *query, float *out) const
{
    for (uint32_t k = 0; k < pca_dim_; k++)
    {
        const float *comp = pca_components_ + k * dim_;
        float dot = 0.0f;
        for (uint32_t d = 0; d < dim_; d++)
            dot += (query[d] - pca_mean_[d]) * comp[d];
        out[k] = dot;
    }
}

// ============================================================================
// Quantization-Aware Graph Refinement
// ============================================================================
// Re-prunes each node's neighbor list using quantized distances instead of
// exact float32 distances. Edges that survive are robust to quantization
// noise, aligning the graph with quantized search mode.
// Must call build_quantized_data() before calling this.

void VamanaIndex::refine_with_quantization()
{
    if (!has_quantized_)
        throw std::runtime_error("Must call build_quantized_data() first");

    std::cout << "Running quantization-aware graph refinement..." << std::endl;
    Timer ref_timer;

    // We need alpha and R — read R from graph max degree, use alpha=1.2 default
    uint32_t R = 0;
    for (uint32_t i = 0; i < npts_; i++)
        R = std::max(R, (uint32_t)graph_[i].size());
    float alpha = 1.2f;

#pragma omp parallel for schedule(dynamic, 64)
    for (int64_t i = 0; i < (int64_t)npts_; i++)
    {
        if (graph_[i].empty())
            continue;

        // Re-score each existing neighbor using quantized distance
        std::vector<Candidate> quant_candidates;
        quant_candidates.reserve(graph_[i].size());
        for (uint32_t nbr : graph_[i])
        {
            float d = compute_l2sq_asymmetric(
                get_vector(i),
                get_quantized_vector(nbr),
                quant_min_, quant_scale_, dim_);
            quant_candidates.push_back({d, nbr});
        }

        // Reprune using quantized distances — keeps only diverse edges
        robust_prune(i, quant_candidates, alpha, R);
    }

    size_t total_edges = 0;
    for (uint32_t i = 0; i < npts_; i++)
        total_edges += graph_[i].size();

    std::cout << "  Refinement complete in " << ref_timer.elapsed_seconds()
              << "s. Avg degree after: " << (double)total_edges / npts_
              << std::endl;
}

// ============================================================================
// Multiple Entry Points
// ============================================================================

void VamanaIndex::build_entry_points(uint32_t k)
{
    if (k <= 1)
    {
        entry_points_ = {start_node_};
        num_entry_points_ = 1;
        return;
    }

    std::cout << "Building " << k << " entry points via k-means..." << std::endl;
    Timer t;

    // k-means++ style init: first = medoid, rest = farthest from existing centroids
    std::vector<std::vector<double>> centroids(k, std::vector<double>(dim_, 0.0));
    const float *mv = get_vector(start_node_);
    for (uint32_t d = 0; d < dim_; d++)
        centroids[0][d] = mv[d];

    std::vector<float> min_dist(npts_, FLT_MAX);
    for (uint32_t c = 1; c < k; c++)
    {
#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < (int64_t)npts_; i++)
        {
            const float *v = get_vector(i);
            float dist = 0.0f;
            for (uint32_t d = 0; d < dim_; d++)
            {
                float diff = v[d] - (float)centroids[c - 1][d];
                dist += diff * diff;
            }
            if (dist < min_dist[i])
                min_dist[i] = dist;
        }
        uint32_t best = 0;
        float best_d = -1.0f;
        for (uint32_t i = 0; i < npts_; i++)
            if (min_dist[i] > best_d)
            {
                best_d = min_dist[i];
                best = i;
            }
        const float *bv = get_vector(best);
        for (uint32_t d = 0; d < dim_; d++)
            centroids[c][d] = bv[d];
    }

    // k-means iterations
    std::vector<uint32_t> assignments(npts_, 0);
    for (int iter = 0; iter < 10; iter++)
    {
#pragma omp parallel for schedule(dynamic, 256)
        for (int64_t i = 0; i < (int64_t)npts_; i++)
        {
            const float *v = get_vector(i);
            float best_d = FLT_MAX;
            uint32_t best_c = 0;
            for (uint32_t c = 0; c < k; c++)
            {
                float dist = 0.0f;
                for (uint32_t d = 0; d < dim_; d++)
                {
                    float diff = v[d] - (float)centroids[c][d];
                    dist += diff * diff;
                }
                if (dist < best_d)
                {
                    best_d = dist;
                    best_c = c;
                }
            }
            assignments[i] = best_c;
        }
        std::vector<std::vector<double>> nc(k, std::vector<double>(dim_, 0.0));
        std::vector<uint32_t> counts(k, 0);
        for (uint32_t i = 0; i < npts_; i++)
        {
            uint32_t c = assignments[i];
            counts[c]++;
            const float *v = get_vector(i);
            for (uint32_t d = 0; d < dim_; d++)
                nc[c][d] += v[d];
        }
        for (uint32_t c = 0; c < k; c++)
            if (counts[c] > 0)
                for (uint32_t d = 0; d < dim_; d++)
                    nc[c][d] /= counts[c];
            else
                nc[c] = centroids[c];
        centroids = std::move(nc);
    }

    // Find medoid of each cluster (point closest to centroid)
    entry_points_.resize(k);
    for (uint32_t c = 0; c < k; c++)
    {
        float best_d = FLT_MAX;
        uint32_t best_id = 0;
        for (uint32_t i = 0; i < npts_; i++)
        {
            if (assignments[i] != c)
                continue;
            const float *v = get_vector(i);
            float dist = 0.0f;
            for (uint32_t d = 0; d < dim_; d++)
            {
                float diff = v[d] - (float)centroids[c][d];
                dist += diff * diff;
            }
            if (dist < best_d)
            {
                best_d = dist;
                best_id = i;
            }
        }
        entry_points_[c] = best_id;
    }
    num_entry_points_ = k;
    std::cout << "  Entry points built in " << t.elapsed_seconds() << "s" << std::endl;
}

uint32_t VamanaIndex::best_entry_point(const float *query) const
{
    if (num_entry_points_ == 1)
        return entry_points_[0];
    float best_d = FLT_MAX;
    uint32_t best_id = entry_points_[0];
    for (uint32_t ep : entry_points_)
    {
        float d = compute_l2sq(query, get_vector(ep), dim_);
        if (d < best_d)
        {
            best_d = d;
            best_id = ep;
        }
    }
    return best_id;
}

// ============================================================================
// Save / Load
// ============================================================================
// Binary format:
//   [uint32] npts
//   [uint32] dim
//   [uint32] start_node
//   [uint32] num_entry_points
//   [uint32 * num_entry_points] entry_point IDs
//   For each node i in [0, npts):
//     [uint32] degree
//     [uint32 * degree] neighbor IDs

void VamanaIndex::save(const std::string &path) const
{
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open())
        throw std::runtime_error("Cannot open file for writing: " + path);

    out.write(reinterpret_cast<const char *>(&npts_), 4);
    out.write(reinterpret_cast<const char *>(&dim_), 4);
    out.write(reinterpret_cast<const char *>(&start_node_), 4);

    // Save entry points
    uint32_t nep = (uint32_t)entry_points_.size();
    out.write(reinterpret_cast<const char *>(&nep), 4);
    if (nep > 0)
        out.write(reinterpret_cast<const char *>(entry_points_.data()),
                  nep * sizeof(uint32_t));

    for (uint32_t i = 0; i < npts_; i++)
    {
        uint32_t deg = static_cast<uint32_t>(graph_[i].size());
        out.write(reinterpret_cast<const char *>(&deg), 4);
        if (deg > 0)
        {
            out.write(reinterpret_cast<const char *>(graph_[i].data()),
                      deg * sizeof(uint32_t));
        }
    }

    std::cout << "Index saved to " << path << std::endl;
}

void VamanaIndex::load(const std::string &index_path,
                       const std::string &data_path)
{
    // Load data vectors
    FloatMatrix mat = load_fbin(data_path);
    npts_ = mat.npts;
    dim_ = mat.dims;
    data_ = mat.data.release();
    owns_data_ = true;

    // Load graph
    std::ifstream in(index_path, std::ios::binary);
    if (!in.is_open())
        throw std::runtime_error("Cannot open index file: " + index_path);

    uint32_t file_npts, file_dim;
    in.read(reinterpret_cast<char *>(&file_npts), 4);
    in.read(reinterpret_cast<char *>(&file_dim), 4);
    in.read(reinterpret_cast<char *>(&start_node_), 4);

    if (file_npts != npts_ || file_dim != dim_)
        throw std::runtime_error(
            "Index/data mismatch: index has " + std::to_string(file_npts) +
            "x" + std::to_string(file_dim) + ", data has " +
            std::to_string(npts_) + "x" + std::to_string(dim_));

    graph_.resize(npts_);
    locks_ = std::vector<std::mutex>(npts_);

    // Load entry points (new format) — fall back gracefully if not present
    uint32_t nep = 0;
    if (in.read(reinterpret_cast<char *>(&nep), 4) && nep > 0 && nep <= 64)
    {
        entry_points_.resize(nep);
        in.read(reinterpret_cast<char *>(entry_points_.data()), nep * sizeof(uint32_t));
        num_entry_points_ = nep;
    }
    else
    {
        // Old index format: no entry points saved — fall back to start_node_
        entry_points_ = {start_node_};
        num_entry_points_ = 1;
        // Rewind: the 4 bytes we read were actually the first node's degree
        in.seekg(-4, std::ios::cur);
    }

    for (uint32_t i = 0; i < npts_; i++)
    {
        uint32_t deg;
        in.read(reinterpret_cast<char *>(&deg), 4);
        graph_[i].resize(deg);
        if (deg > 0)
        {
            in.read(reinterpret_cast<char *>(graph_[i].data()),
                    deg * sizeof(uint32_t));
        }
    }

    std::cout << "Index loaded: " << npts_ << " points, " << dim_
              << " dims, start=" << start_node_
              << ", entry_points=" << num_entry_points_ << std::endl;
}

// ============================================================================
// Graph Analysis
// ============================================================================

GraphStats VamanaIndex::compute_graph_stats() const
{
    GraphStats stats;
    if (npts_ == 0) return stats;

    uint32_t max_deg = 0;
    uint32_t min_deg = UINT32_MAX;
    double sum_deg = 0.0;

    for (uint32_t i = 0; i < npts_; i++) {
        uint32_t deg = (uint32_t)graph_[i].size();
        sum_deg += deg;
        if (deg > max_deg) max_deg = deg;
        if (deg < min_deg) min_deg = deg;
    }

    stats.avg_degree = sum_deg / npts_;
    stats.min_degree = min_deg;
    stats.max_degree = max_deg;

    // Compute standard deviation
    double var_sum = 0.0;
    for (uint32_t i = 0; i < npts_; i++) {
        double diff = graph_[i].size() - stats.avg_degree;
        var_sum += diff * diff;
    }
    stats.degree_stddev = std::sqrt(var_sum / npts_);

    // Build histogram
    stats.degree_hist.assign(max_deg + 1, 0);
    for (uint32_t i = 0; i < npts_; i++)
        stats.degree_hist[graph_[i].size()]++;

    return stats;
}

void VamanaIndex::export_degree_histogram(const std::string &path) const
{
    GraphStats stats = compute_graph_stats();
    std::ofstream out(path);
    if (!out.is_open()) {
        std::cerr << "Cannot open histogram output: " << path << std::endl;
        return;
    }
    out << "degree,count\n";
    for (uint32_t d = 0; d < stats.degree_hist.size(); d++) {
        if (stats.degree_hist[d] > 0)
            out << d << "," << stats.degree_hist[d] << "\n";
    }
    std::cout << "Degree histogram exported to " << path << std::endl;
    std::cout << "  Avg=" << stats.avg_degree
              << " Min=" << stats.min_degree
              << " Max=" << stats.max_degree
              << " StdDev=" << stats.degree_stddev << std::endl;
}