#!/usr/bin/env bash
# SIFT1M benchmark: builds the three reference indexes (baseline R=32,
# improved R=32, improved R=64) and runs every search mode three times,
# single-threaded (OMP_NUM_THREADS=1). Tables go to logs/summary.md.
# Converts SIFT1M/*.fvecs to tmp/*.fbin on first run (scripts/convert_vecs.py).
# Optional last step benchmarks the gpu/ tools when PyTorch sees a CUDA GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
cmake -S . -B build_simd -DCMAKE_BUILD_TYPE=Release -DGRAPHANN_SIMD_REDUCTION=ON && cmake --build build_simd -j --target search_index
mkdir -p logs tmp
T=tmp
# Convert the texmex files in SIFT1M/ if the binary copies are not there yet.
[ -f $T/sift_base.fbin ]  || python3 scripts/convert_vecs.py SIFT1M/sift_base.fvecs $T/sift_base.fbin
[ -f $T/sift_query.fbin ] || python3 scripts/convert_vecs.py SIFT1M/sift_query.fvecs $T/sift_query.fbin
[ -f $T/sift_gt.ibin ]    || python3 scripts/convert_vecs.py SIFT1M/sift_groundtruth.ivecs $T/sift_gt.ibin
COMMON="--data $T/sift_base.fbin --L 75 --alpha 1.2 --gamma 1.5 --single_pass"
build/build_index $COMMON --R 32 --baseline --output $T/idx_base_r32.bin > logs/build_base_r32.log 2>&1
build/build_index $COMMON --R 32            --output $T/idx_impr_r32.bin > logs/build_impr_r32.log 2>&1
build/build_index $COMMON --R 64            --output $T/idx_impr_r64.bin > logs/build_impr_r64.log 2>&1
S="--data $T/sift_base.fbin --queries $T/sift_query.fbin --gt $T/sift_gt.ibin --K 10 --L 10,20,30,50,75,100,150,200"
export OMP_NUM_THREADS=1
for rep in 1 2 3; do
  for idx in base_r32 impr_r32 impr_r64; do
    build/search_index      --index $T/idx_$idx.bin $S > logs/search_${idx}_exact_rep$rep.log 2>&1
    build_simd/search_index --index $T/idx_$idx.bin $S > logs/search_${idx}_exactSIMD_rep$rep.log 2>&1
  done
  build/search_index --index $T/idx_impr_r64.bin $S --quantized           > logs/search_impr_r64_adc_rep$rep.log 2>&1
  build/search_index --index $T/idx_impr_r64.bin $S --quantized --dynamic > logs/search_impr_r64_adcdyn_rep$rep.log 2>&1
  build/search_index --index $T/idx_base_r32.bin $S --quantized           > logs/search_base_r32_adc_rep$rep.log 2>&1
done
python3 scripts/summarize_search_logs.py | tee logs/summary.md

# Optional: GPU tools (needs a CUDA build of PyTorch). Skipped if no GPU.
if python3 -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  for w in 1 4; do
    python3 gpu/batched_search.py --index $T/idx_impr_r64.bin --data $T/sift_base.fbin \
      --queries $T/sift_query.fbin --gt $T/sift_gt.ibin --L 10,20,30,50,75,100,150,200 \
      --batch 1024 --width $w > logs/gpu_search_impr_r64_w$w.log 2>&1
  done
  python3 gpu/bruteforce_knn.py --data $T/sift_base.fbin --queries $T/sift_query.fbin \
    --K 10 --check $T/sift_gt.ibin > logs/gpu_bruteforce.log 2>&1
  cat logs/gpu_search_impr_r64_w*.log logs/gpu_bruteforce.log
else
  echo "No CUDA GPU visible to PyTorch; skipped gpu/ benchmarks."
fi
