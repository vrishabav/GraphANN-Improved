#!/usr/bin/env bash
# run_ablation.sh — Runs all 16 ablation conditions for Section 2 improvements.
# For each condition: builds index, exports degree histogram, runs search at
# multiple L values, and appends results to ablation_results.tsv.
#
# Usage: ./scripts/run_ablation.sh [--small]
#   --small  Use 100K subset for fast iteration (requires sift_base_100k.fbin)
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/build"
DATA="$ROOT/tmp"

BASE_FBIN="${DATA}/sift_base.fbin"
QUERY_FBIN="${DATA}/sift_query.fbin"
GT_IBIN="${DATA}/sift_gt.ibin"
INDEX_DIR="${DATA}/ablation_indices"
HIST_DIR="${DATA}/ablation_histograms"
RESULTS="${DATA}/ablation_results.tsv"

mkdir -p "$INDEX_DIR" "$HIST_DIR"

# Use small subset if requested
if [[ "${1:-}" == "--small" ]]; then
    BASE_FBIN="${DATA}/sift_base_100k.fbin"
    echo "Using 100K subset: $BASE_FBIN"
fi

# Header for results TSV
echo -e "medoid\ttwo_pass\trandom_init\tstop_degree\tbuild_time_s\tavg_degree\tstddev_degree\tL\trecall\tavg_latency_us\tp99_latency_us\tdist_cmps" > "$RESULTS"

run_condition() {
    local medoid=$1 two_pass=$2 random_init=$3 strict=$4
    local tag="m${medoid}_tp${two_pass}_ri${random_init}_sd${strict}"
    local index="${INDEX_DIR}/index_${tag}.bin"
    local hist="${HIST_DIR}/hist_${tag}.csv"

    echo ""
    echo "======================================================"
    echo "Condition: medoid=$medoid two_pass=$two_pass random_init=$random_init strict=$strict"
    echo "======================================================"

    # Build flags
    local flags="--R 32 --L 75 --alpha 1.2 --gamma 1.5"
    [[ $medoid      -eq 1 ]] && flags+=" --medoid"
    [[ $two_pass    -eq 1 ]] && flags+=" --two-pass"
    [[ $random_init -eq 1 ]] && flags+=" --random-init"
    [[ $strict      -eq 1 ]] && flags+=" --strict-degree"

    # Build and capture build time
    local t0=$(date +%s%N)
    "$BUILD/build_index_v2" --data "$BASE_FBIN" --output "$index" \
        $flags --export-hist "$hist" 2>&1 | tee "${INDEX_DIR}/build_${tag}.log"
    local t1=$(date +%s%N)
    local build_time_s=$(echo "scale=2; ($t1 - $t0) / 1000000000" | bc)

    # Extract avg/stddev degree from log
    local avg_deg=$(grep "Average out-degree" "${INDEX_DIR}/build_${tag}.log" | awk '{print $NF}')
    local std_deg=$(grep "Std-dev out-degree"  "${INDEX_DIR}/build_${tag}.log" | awk '{print $NF}')

    # Search at multiple L values — exact float32 (no proposals A/B/C)
    local search_log="${INDEX_DIR}/search_${tag}.log"
    "$BUILD/search_index" \
        --index "$index" \
        --data  "$BASE_FBIN" \
        --queries "$QUERY_FBIN" \
        --gt "$GT_IBIN" \
        --K 10 --L "10,20,50,75,100,150,200" \
        2>&1 | tee "$search_log"

    # Parse search results and append to TSV
    # Search output has lines like: "      75        0.9818         1995.4          627.8          2672.4"
    while IFS= read -r line; do
        if [[ "$line" =~ ^[[:space:]]+([0-9]+)[[:space:]] ]]; then
            local L recall _ lat_avg lat_p99 dist_cmps
            read L recall dist_cmps lat_avg lat_p99 <<< "$line"
            echo -e "${medoid}\t${two_pass}\t${random_init}\t${strict}\t${build_time_s}\t${avg_deg}\t${std_deg}\t${L}\t${recall}\t${lat_avg}\t${lat_p99}\t${dist_cmps}" \
                >> "$RESULTS"
        fi
    done < "$search_log"

    echo "Done: $tag (build=${build_time_s}s)"
}

# ── Ensure project is built ───────────────────────────────────────────────────
echo "=== Building project ==="
mkdir -p "$BUILD"
pushd "$BUILD" > /dev/null
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -march=native" > /dev/null
make -j"$(nproc)" 2>&1 | grep -E "make\[|Error|error:" || true
popd > /dev/null

# ── Run all 16 conditions ─────────────────────────────────────────────────────
# medoid  two_pass  random_init  strict_degree
run_condition 0 0 0 0   # Baseline
run_condition 1 0 0 0   # Medoid only
run_condition 0 1 0 0   # Two-pass only
run_condition 0 0 1 0   # Random init only
run_condition 0 0 0 1   # Strict degree only
run_condition 1 1 0 0   # Medoid + Two-pass
run_condition 1 0 1 0   # Medoid + Random init
run_condition 1 0 0 1   # Medoid + Strict degree
run_condition 0 1 1 0   # Two-pass + Random init
run_condition 0 1 0 1   # Two-pass + Strict degree
run_condition 0 0 1 1   # Random init + Strict degree
run_condition 1 1 1 0   # Medoid + Two-pass + Random init
run_condition 1 1 0 1   # Medoid + Two-pass + Strict degree
run_condition 1 0 1 1   # Medoid + Random init + Strict degree
run_condition 0 1 1 1   # Two-pass + Random init + Strict degree
run_condition 1 1 1 1   # ALL FOUR (paper-compliant)

echo ""
echo "=== Ablation complete. Results in $RESULTS ==="
echo "Conditions: 16 | L values: 7 | Total rows: $((16 * 7))"
