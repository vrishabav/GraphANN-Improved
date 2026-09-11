#include "hnsw_index.h"
#include "io_utils.h"
#include "timer.h"

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <string>
#include <sstream>
#include <vector>
#include <cstdlib>

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " --index <hnsw_index_path>"
              << " --data <fbin_path>"
              << " --queries <query_fbin_path>"
              << " --gt <ground_truth_ibin_path>"
              << " --K <num_neighbors>"
              << " --ef <comma_separated_ef_values>"
              << std::endl;
    std::cerr << "\nExample:" << std::endl;
    std::cerr << "  " << prog
              << " --index tmp/sift_hnsw_m16.bin"
              << " --data tmp/sift_base.fbin"
              << " --queries tmp/sift_query.fbin"
              << " --gt tmp/sift_gt.ibin"
              << " --K 10 --ef 10,20,50,100,200"
              << std::endl;
}

// Parse comma-separated ef values like "10,20,50,100"
static std::vector<uint32_t> parse_ef_values(const std::string& s) {
    std::vector<uint32_t> values;
    std::istringstream stream(s);
    std::string token;
    while (std::getline(stream, token, ','))
        values.push_back(std::atoi(token.c_str()));
    std::sort(values.begin(), values.end());
    return values;
}

// Compute recall@K: fraction of true top-K neighbors found in result
static double compute_recall(const std::vector<uint32_t>& result,
                              const uint32_t* gt, uint32_t K) {
    uint32_t found = 0;
    for (uint32_t i = 0; i < K && i < (uint32_t)result.size(); i++) {
        for (uint32_t j = 0; j < K; j++) {
            if (result[i] == gt[j]) {
                found++;
                break;
            }
        }
    }
    return (double)found / K;
}

int main(int argc, char** argv) {
    std::string index_path, data_path, query_path, gt_path, ef_str;
    uint32_t K = 10;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if      (arg == "--index"   && i + 1 < argc) index_path = argv[++i];
        else if (arg == "--data"    && i + 1 < argc) data_path  = argv[++i];
        else if (arg == "--queries" && i + 1 < argc) query_path = argv[++i];
        else if (arg == "--gt"      && i + 1 < argc) gt_path    = argv[++i];
        else if (arg == "--K"       && i + 1 < argc) K          = std::atoi(argv[++i]);
        else if (arg == "--ef"      && i + 1 < argc) ef_str     = argv[++i];
        else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (index_path.empty() || data_path.empty() || query_path.empty() ||
        gt_path.empty() || ef_str.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    std::vector<uint32_t> ef_values = parse_ef_values(ef_str);
    if (ef_values.empty()) {
        std::cerr << "Error: no ef values provided." << std::endl;
        return 1;
    }

    // --- Load index ---
    std::cout << "Loading HNSW index..." << std::endl;
    HNSWIndex index;
    index.load(index_path, data_path);

    // --- Load queries ---
    std::cout << "Loading queries from " << query_path << "..." << std::endl;
    FloatMatrix queries = load_fbin(query_path);
    std::cout << "  Queries: " << queries.npts << " x " << queries.dims << std::endl;

    if (queries.dims != index.get_dim()) {
        std::cerr << "Error: query dimension (" << queries.dims
                  << ") != index dimension (" << index.get_dim() << ")" << std::endl;
        return 1;
    }

    // --- Load ground truth ---
    std::cout << "Loading ground truth from " << gt_path << "..." << std::endl;
    IntMatrix gt = load_ibin(gt_path);
    std::cout << "  Ground truth: " << gt.npts << " x " << gt.dims << std::endl;

    if (gt.npts != queries.npts) {
        std::cerr << "Error: ground truth rows (" << gt.npts
                  << ") != number of queries (" << queries.npts << ")" << std::endl;
        return 1;
    }
    if (gt.dims < K) {
        std::cerr << "Warning: ground truth has " << gt.dims
                  << " neighbors per query but K=" << K << std::endl;
        K = gt.dims;
    }

    uint32_t nq = queries.npts;

    // --- Run search for each ef value ---
    std::cout << "\n=== HNSW Search Results (K=" << K
              << ", M=" << index.get_M()
              << ", efC=" << index.get_efC()
              << ") ===" << std::endl;
    std::cout << std::setw(8)  << "ef"
              << std::setw(14) << "Recall@" + std::to_string(K)
              << std::setw(16) << "Avg Dist Cmps"
              << std::setw(18) << "Avg Latency (us)"
              << std::setw(18) << "P99 Latency (us)"
              << std::endl;
    std::cout << std::string(74, '-') << std::endl;

    for (uint32_t ef : ef_values) {
        std::vector<double>   recalls(nq);
        std::vector<uint32_t> dist_cmps(nq);
        std::vector<double>   latencies(nq);

        // Sequential search (HNSW search is not thread-safe during build,
        // but after build completes the graph is read-only so parallel is fine)
#pragma omp parallel for schedule(dynamic, 16)
        for (int32_t q = 0; q < (int32_t)nq; q++) {
            HNSWSearchResult res = index.search(queries.row(q), K, ef);
            recalls[q]   = compute_recall(res.ids, gt.row(q), K);
            dist_cmps[q] = res.dist_cmps;
            latencies[q] = res.latency_us;
        }

        // Aggregate statistics
        double avg_recall = std::accumulate(recalls.begin(), recalls.end(), 0.0) / nq;
        double avg_cmps   = (double)std::accumulate(dist_cmps.begin(), dist_cmps.end(), 0ULL) / nq;
        double avg_lat    = std::accumulate(latencies.begin(), latencies.end(), 0.0) / nq;

        // P99 latency
        std::sort(latencies.begin(), latencies.end());
        double p99_lat = latencies[(size_t)(0.99 * nq)];

        std::cout << std::setw(8)  << ef
                  << std::setw(14) << std::fixed << std::setprecision(4) << avg_recall
                  << std::setw(16) << std::fixed << std::setprecision(1) << avg_cmps
                  << std::setw(18) << std::fixed << std::setprecision(1) << avg_lat
                  << std::setw(18) << std::fixed << std::setprecision(1) << p99_lat
                  << std::endl;
    }

    std::cout << "\nDone." << std::endl;
    return 0;
}
