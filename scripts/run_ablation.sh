#!/usr/bin/env bash
# run_ablation.sh - build-configuration ablation on SIFT1M.
# Every combination of
#   build:   baseline (--baseline: random start, empty graph, top-L prune)
#            improved (medoid, random R-regular init, full visited set)
#   passes:  single (--single_pass) | two-pass (alpha 1.0 then 1.2)
#   degree:  R=32 | R=64
# = 8 conditions. Each builds an index, searches it single-threaded at several
# L values and appends one row per L to ablation_results.tsv.
#
# Usage: ./scripts/run_ablation.sh [--small]
#   --small  use a 100K subset (tmp/sift_base_100k.fbin) for fast iteration
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/build"
DATA="$ROOT/tmp"
BASE_FBIN="${DATA}/sift_base.fbin"
QUERY_FBIN="${DATA}/sift_query.fbin"
GT_IBIN="${DATA}/sift_gt.ibin"
INDEX_DIR="${DATA}/ablation_indices"
RESULTS="${DATA}/ablation_results.tsv"
mkdir -p "$INDEX_DIR"

if [[ "${1:-}" == "--small" ]]; then
    BASE_FBIN="${DATA}/sift_base_100k.fbin"
    echo "Using 100K subset: $BASE_FBIN (ground truth must match this subset)"
fi

echo -e "build\ttwo_pass\tR\tbuild_time_s\tavg_degree\tL\trecall\tavg_latency_us\tp99_latency_us\tdist_cmps" > "$RESULTS"

run_condition() {
    local build=$1 two_pass=$2 R=$3
    local tag="${build}_tp${two_pass}_r${R}"
    local index="${INDEX_DIR}/index_${tag}.bin"
    local flags="--R $R --L 75 --alpha 1.2 --gamma 1.5"
    [[ $build == baseline ]] && flags+=" --baseline"
    [[ $two_pass -eq 0 ]] && flags+=" --single_pass"

    echo "=== $tag ==="
    "$BUILD/build_index" --data "$BASE_FBIN" --output "$index" $flags > "${INDEX_DIR}/build_${tag}.log" 2>&1
    local build_time_s avg_deg
    build_time_s=$(tr '\r' '\n' < "${INDEX_DIR}/build_${tag}.log" | grep "Total build time" | awk '{print $(NF-1)}')
    avg_deg=$(tr '\r' '\n' < "${INDEX_DIR}/build_${tag}.log" | grep "Average out-degree" | tail -1 | awk '{print $NF}')

    local search_log="${INDEX_DIR}/search_${tag}.log"
    OMP_NUM_THREADS=1 "$BUILD/search_index" --index "$index" --data "$BASE_FBIN" \
        --queries "$QUERY_FBIN" --gt "$GT_IBIN" --K 10 --L "10,20,30,50,75,100,150,200" > "$search_log" 2>&1

    # rows look like: "      75        0.9818         1995.4          627.8          2672.4"
    while IFS= read -r line; do
        if [[ "$line" =~ ^[[:space:]]+([0-9]+)[[:space:]]+0\. ]]; then
            read -r L recall dist_cmps lat_avg lat_p99 <<< "$line"
            echo -e "${build}\t${two_pass}\t${R}\t${build_time_s}\t${avg_deg}\t${L}\t${recall}\t${lat_avg}\t${lat_p99}\t${dist_cmps}" >> "$RESULTS"
        fi
    done < "$search_log"
    echo "done: $tag (build ${build_time_s}s, avg degree ${avg_deg})"
}

echo "=== Building project ==="
cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build "$BUILD" -j > /dev/null

for build in baseline improved; do
  for two_pass in 0 1; do
    for R in 32 64; do
      run_condition "$build" "$two_pass" "$R"
    done
  done
done

echo "=== Ablation complete. Results in $RESULTS (8 conditions x 8 L values) ==="
