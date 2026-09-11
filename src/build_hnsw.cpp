#include "hnsw_index.h"
#include "timer.h"

#include <iostream>
#include <string>
#include <cstdlib>

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " --data <fbin_path>"
              << " --output <index_path>"
              << " [--M <max_connections=16>]"
              << " [--efC <efConstruction=200>]"
              << " [--seed <random_seed=42>]"
              << std::endl;
    std::cerr << "\nHNSW Parameters:" << std::endl;
    std::cerr << "  --M    : Max connections per layer (layer 0 uses 2*M). "
                 "Typical: 8, 16, 32." << std::endl;
    std::cerr << "  --efC  : Beam width during construction. "
                 "Higher = better quality, slower build. Typical: 100-400." << std::endl;
}

int main(int argc, char** argv) {
    std::string data_path, output_path;
    uint32_t M              = 16;
    uint32_t efConstruction = 200;
    uint32_t seed           = 42;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if      (arg == "--data"   && i + 1 < argc) data_path   = argv[++i];
        else if (arg == "--output" && i + 1 < argc) output_path = argv[++i];
        else if (arg == "--M"      && i + 1 < argc) M           = std::atoi(argv[++i]);
        else if (arg == "--efC"    && i + 1 < argc) efConstruction = std::atoi(argv[++i]);
        else if (arg == "--seed"   && i + 1 < argc) seed        = std::atoi(argv[++i]);
        else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (data_path.empty() || output_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    std::cout << "=== HNSW Index Builder ===" << std::endl;
    std::cout << "Parameters:" << std::endl;
    std::cout << "  M              = " << M              << std::endl;
    std::cout << "  M0 (layer 0)   = " << 2 * M          << std::endl;
    std::cout << "  efConstruction = " << efConstruction << std::endl;
    std::cout << "  seed           = " << seed           << std::endl;

    HNSWIndex index;

    Timer total_timer;
    index.build(data_path, M, efConstruction, seed);
    double total_time = total_timer.elapsed_seconds();

    std::cout << "\nTotal build time: " << total_time << " seconds" << std::endl;

    index.save(output_path);
    std::cout << "Done." << std::endl;
    return 0;
}
