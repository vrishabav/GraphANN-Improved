$ErrorActionPreference = "Stop"

$base    = $PSScriptRoot
$build   = "$base\build"
$tmp     = "$base\tmp"
$data    = "$tmp\sift_base.fbin"
$queries = "$tmp\sift_query.fbin"
$gt      = "$tmp\sift_gt.ibin"

Write-Host "============================================================"
Write-Host "HNSW + SOFT-PRUNE BENCHMARK SUITE"
Write-Host "============================================================"

# ---- HNSW M=16 efC=100 ----
Write-Host "`n[1/6] Building HNSW M=16 efC=100..."
& "$build\build_hnsw.exe" --data $data --output "$tmp\hnsw_m16_efc100.bin" --M 16 --efC 100
if ($LASTEXITCODE -ne 0) { throw "build_hnsw M=16 efC=100 failed" }

Write-Host "`n[1/6] Searching HNSW M=16 efC=100..."
& "$build\search_hnsw.exe" --index "$tmp\hnsw_m16_efc100.bin" --data $data --queries $queries --gt $gt --K 10 --ef "10,20,50,100,200,400"
if ($LASTEXITCODE -ne 0) { throw "search_hnsw M=16 efC=100 failed" }

# ---- HNSW M=16 efC=200 ----
Write-Host "`n[2/6] Building HNSW M=16 efC=200..."
& "$build\build_hnsw.exe" --data $data --output "$tmp\hnsw_m16_efc200.bin" --M 16 --efC 200
if ($LASTEXITCODE -ne 0) { throw "build_hnsw M=16 efC=200 failed" }

Write-Host "`n[2/6] Searching HNSW M=16 efC=200..."
& "$build\search_hnsw.exe" --index "$tmp\hnsw_m16_efc200.bin" --data $data --queries $queries --gt $gt --K 10 --ef "10,20,50,100,200,400"
if ($LASTEXITCODE -ne 0) { throw "search_hnsw M=16 efC=200 failed" }

# ---- HNSW M=32 efC=200 ----
Write-Host "`n[3/6] Building HNSW M=32 efC=200..."
& "$build\build_hnsw.exe" --data $data --output "$tmp\hnsw_m32_efc200.bin" --M 32 --efC 200
if ($LASTEXITCODE -ne 0) { throw "build_hnsw M=32 efC=200 failed" }

Write-Host "`n[3/6] Searching HNSW M=32 efC=200..."
& "$build\search_hnsw.exe" --index "$tmp\hnsw_m32_efc200.bin" --data $data --queries $queries --gt $gt --K 10 --ef "10,20,50,100,200,400"
if ($LASTEXITCODE -ne 0) { throw "search_hnsw M=32 efC=200 failed" }

# ---- Vamana best config (R=64, existing index_improved.bin) ----
Write-Host "`n[4/6] Searching Vamana best config (R=64, quantized ADC)..."
& "$build\search_index.exe" --index "$tmp\index_improved.bin" --data $data --queries $queries --gt $gt --K 10 --L "10,20,50,75,100,200" --quantized
if ($LASTEXITCODE -ne 0) { throw "search_index Vamana R=64 failed" }

# ---- Vamana standard R=32 single-pass (baseline for soft-prune comparison) ----
Write-Host "`n[5/6] Building Vamana standard R=32 single-pass..."
& "$build\build_index.exe" --data $data --output "$tmp\index_standard_r32.bin" --R 32 --L 75 --alpha 1.2 --gamma 1.5 --single_pass
if ($LASTEXITCODE -ne 0) { throw "build_index standard R=32 failed" }

Write-Host "`n[5/6] Searching Vamana standard R=32..."
& "$build\search_index.exe" --index "$tmp\index_standard_r32.bin" --data $data --queries $queries --gt $gt --K 10 --L "10,20,50,75,100,200"
if ($LASTEXITCODE -ne 0) { throw "search_index standard R=32 failed" }

# ---- Vamana soft diversity pruning R=32 single-pass ----
Write-Host "`n[6/6] Building Vamana soft diversity pruning R=32 single-pass (beta=0.5)..."
& "$build\build_index.exe" --data $data --output "$tmp\index_soft_prune.bin" --R 32 --L 75 --alpha 1.2 --gamma 1.5 --single_pass --soft_prune --soft_beta 0.5
if ($LASTEXITCODE -ne 0) { throw "build_index soft_prune failed" }

Write-Host "`n[6/6] Searching Vamana soft diversity pruning R=32..."
& "$build\search_index.exe" --index "$tmp\index_soft_prune.bin" --data $data --queries $queries --gt $gt --K 10 --L "10,20,50,75,100,200"
if ($LASTEXITCODE -ne 0) { throw "search_index soft_prune failed" }

Write-Host "`n============================================================"
Write-Host "ALL BENCHMARKS COMPLETE"
Write-Host "============================================================"
