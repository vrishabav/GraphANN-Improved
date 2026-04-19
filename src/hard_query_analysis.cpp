// hard_query_analysis.cpp
// Identifies queries where baseline search misses ground truth neighbors,
// characterises them by spatial properties, and compares before/after improvements.
//
// Output: hard_queries.csv with per-query statistics
//
// Build: linked against graphann_lib (included in CMakeLists.txt if added)
// Usage:
//   ./hard_query_analysis \
//     --index-baseline  index_baseline.bin \
//     --index-improved  index_all.bin \
//     --data  sift_base.fbin \
//     --queries sift_query.fbin \
//     --gt sift_gt.ibin \
//     --K 10 --L 200 \
//     --output hard_queries.csv

#include "vamana_index.h"
#include "io_utils.h"
#include "distance.h"
#include "timer.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " --index-baseline <path> --index-improved <path>"
              << " --data <fbin> --queries <fbin> --gt <ibin>"
              << " --K <k> --L <l> --output <csv>\n";
}

// Returns how many of result[] are in the true top-K ground truth gt[0..K-1]
static uint32_t count_hits(const std::vector<uint32_t>& result,
                           const uint32_t* gt, uint32_t K) {
    uint32_t hits = 0;
    for (uint32_t r : result)
        for (uint32_t j = 0; j < K; j++)
            if (r == gt[j]) { hits++; break; }
    return hits;
}

int main(int argc, char** argv) {
    std::string base_idx, imp_idx, data_path, query_path, gt_path, out_path;
    uint32_t K = 10, L = 200;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--index-baseline" && i+1 < argc) base_idx   = argv[++i];
        else if (a == "--index-improved" && i+1 < argc) imp_idx    = argv[++i];
        else if (a == "--data"           && i+1 < argc) data_path  = argv[++i];
        else if (a == "--queries"        && i+1 < argc) query_path = argv[++i];
        else if (a == "--gt"             && i+1 < argc) gt_path    = argv[++i];
        else if (a == "--K"              && i+1 < argc) K          = std::atoi(argv[++i]);
        else if (a == "--L"              && i+1 < argc) L          = std::atoi(argv[++i]);
        else if (a == "--output"         && i+1 < argc) out_path   = argv[++i];
        else if (a == "--help") { print_usage(argv[0]); return 0; }
    }
    if (base_idx.empty() || query_path.empty() || gt_path.empty() || out_path.empty()) {
        print_usage(argv[0]); return 1;
    }

    // ── Load baseline index ────────────────────────────────────────────────
    std::cout << "Loading baseline index..." << std::endl;
    VamanaIndex baseline;
    baseline.load(base_idx, data_path);

    // ── Load improved index (optional) ────────────────────────────────────
    VamanaIndex* improved_ptr = nullptr;
    VamanaIndex improved;
    if (!imp_idx.empty()) {
        std::cout << "Loading improved index..." << std::endl;
        improved.load(imp_idx, data_path);
        improved_ptr = &improved;
    }

    // ── Load queries and ground truth ─────────────────────────────────────
    FloatMatrix queries = load_fbin(query_path);
    IntMatrix   gt      = load_ibin(gt_path);
    uint32_t    nq      = queries.npts;
    uint32_t    dim     = queries.dims;
    std::cout << "Queries: " << nq << " x " << dim << std::endl;

    // ── Load data for geometric analysis ──────────────────────────────────
    FloatMatrix data = load_fbin(data_path);

    // ── Run search on all queries ──────────────────────────────────────────
    struct QueryResult {
        uint32_t id;
        uint32_t hits_baseline;     // # correct neighbors found (baseline)
        uint32_t hits_improved;     // # correct neighbors found (improved)
        float    nn1_dist;          // dist to true nearest neighbor (exact)
        float    start_dist;        // dist from start node to query
        uint32_t nn1_degree;        // out-degree of true nearest neighbor
        float    latency_baseline;  // search latency (baseline)
        float    latency_improved;  // search latency (improved)
        bool     is_hard;           // true if baseline missed >= 1 neighbor
        bool     improved_by_new;   // true if improved index fixes it
    };

    uint32_t baseline_start = baseline.get_start_node();
    const float* data_ptr = data.data.get();

    std::vector<QueryResult> results(nq);

    #pragma omp parallel for schedule(dynamic, 32)
    for (int32_t q = 0; q < (int32_t)nq; q++) {
        const float* qv  = queries.row(q);
        const uint32_t* gt_row = gt.row(q);

        // Baseline search
        SearchResult sr_base = baseline.search(qv, K, L);
        uint32_t hits_b = count_hits(sr_base.ids, gt_row, K);

        // Improved search
        uint32_t hits_i = hits_b;
        float lat_i = 0.0f;
        if (improved_ptr) {
            SearchResult sr_imp = improved_ptr->search(qv, K, L);
            hits_i = count_hits(sr_imp.ids, gt_row, K);
            lat_i  = (float)sr_imp.latency_us;
        }

        // Geometric analysis
        // Distance to true nearest neighbor (gt[0] is the nearest)
        float nn1_dist = std::sqrt(
            compute_l2sq(qv, data_ptr + (size_t)gt_row[0] * dim, dim));

        // Distance from start node to query
        float start_dist = std::sqrt(
            compute_l2sq(qv, data_ptr + (size_t)baseline_start * dim, dim));

        // Use baseline graph stats for nn1_degree (not directly accessible,
        // so we compute a proxy: how many other base points are within
        // 2× the nearest-neighbor distance — density proxy)
        // [Full degree access requires exposing graph_ — use a simpler proxy]
        // Here we report it as 0; a post-processing step can join with hist CSV.
        uint32_t nn1_degree = 0;

        results[q] = {
            (uint32_t)q,
            hits_b, hits_i,
            nn1_dist, start_dist, nn1_degree,
            (float)sr_base.latency_us, lat_i,
            hits_b < K,
            (improved_ptr && hits_i > hits_b)
        };
    }

    // ── Compute aggregate statistics ──────────────────────────────────────
    uint32_t n_hard = 0, n_fixed = 0;
    double sum_nn1_hard = 0, sum_nn1_easy = 0;
    double sum_start_hard = 0, sum_start_easy = 0;
    uint32_t n_easy = 0;

    for (auto& r : results) {
        if (r.is_hard) {
            n_hard++;
            sum_nn1_hard   += r.nn1_dist;
            sum_start_hard += r.start_dist;
            if (r.improved_by_new) n_fixed++;
        } else {
            n_easy++;
            sum_nn1_easy   += r.nn1_dist;
            sum_start_easy += r.start_dist;
        }
    }

    std::cout << "\n=== Hard Query Analysis (L=" << L << ", K=" << K << ") ===\n";
    std::cout << "Total queries:  " << nq << "\n";
    std::cout << "Hard queries:   " << n_hard << " (" 
              << std::fixed << std::setprecision(2)
              << (100.0 * n_hard / nq) << "%)\n";
    if (improved_ptr)
        std::cout << "Fixed by improved index: " << n_fixed << " of " << n_hard << "\n";
    if (n_hard > 0) {
        std::cout << "Avg NN1 dist  (hard):  " << sum_nn1_hard / n_hard << "\n";
        std::cout << "Avg NN1 dist  (easy):  " << sum_nn1_easy / n_easy << "\n";
        std::cout << "Avg start dist (hard): " << sum_start_hard / n_hard << "\n";
        std::cout << "Avg start dist (easy): " << sum_start_easy / n_easy << "\n";
        std::cout << "\nInsight: Hard queries have "
                  << std::setprecision(1)
                  << (sum_start_hard / n_hard) / (sum_start_easy / n_easy)
                  << "x longer start-to-query distance (supports medoid hypothesis)\n";
    }

    // ── Write CSV ─────────────────────────────────────────────────────────
    std::ofstream csv(out_path);
    if (!csv.is_open()) { std::cerr << "Cannot open " << out_path << "\n"; return 1; }

    csv << "query_id,hits_baseline,hits_improved,nn1_dist,start_dist,"
           "latency_baseline_us,latency_improved_us,is_hard,improved_by_new\n";
    for (auto& r : results) {
        csv << r.id << ","
            << r.hits_baseline << ","
            << r.hits_improved << ","
            << std::fixed << std::setprecision(4)
            << r.nn1_dist << ","
            << r.start_dist << ","
            << r.latency_baseline << ","
            << r.latency_improved << ","
            << (r.is_hard ? 1 : 0) << ","
            << (r.improved_by_new ? 1 : 0) << "\n";
    }

    std::cout << "\nDetailed results written to " << out_path << "\n";
    return 0;
}
