$ErrorActionPreference = "Continue"

$base    = $PSScriptRoot
$build   = "$base\build"
$tmp     = "$base\tmp"
$logs    = "$tmp\logs"
$data    = "$tmp\sift_base.fbin"
$queries = "$tmp\sift_query.fbin"
$gt      = "$tmp\sift_gt.ibin"

if (-not (Test-Path $logs)) { New-Item -ItemType Directory -Path $logs | Out-Null }

function Run-Step {
    param([string]$Label, [string]$LogFile, [scriptblock]$Cmd)
    Write-Host ""
    Write-Host "===========================================================" -ForegroundColor Cyan
    Write-Host "  $Label" -ForegroundColor Cyan
    Write-Host "===========================================================" -ForegroundColor Cyan
    $start = Get-Date
    & $Cmd 2>&1 | Tee-Object -FilePath $LogFile
    $elapsed = [math]::Round(((Get-Date) - $start).TotalSeconds, 1)
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[DONE] $Label  (wall: ${elapsed}s)" -ForegroundColor Green
    } else {
        Write-Host "[FAILED] $Label (exit $LASTEXITCODE)" -ForegroundColor Red
    }
}

Write-Host "===========================================================" -ForegroundColor Yellow
Write-Host "  SUPER-FAST BENCHMARK: HNSW M=16 vs Vamana Baselines" -ForegroundColor Yellow
Write-Host "===========================================================" -ForegroundColor Yellow

# Build HNSW M=16 with lower efC for speed (should take ~8-10 mins)
Run-Step "BUILD HNSW M=16 efC=40" "$logs\hnsw_m16_efc40_build.log" {
    & "$build\build_hnsw.exe" --data $data --output "$tmp\hnsw_m16_efc40.bin" --M 16 --efC 40
}
Run-Step "SEARCH HNSW M=16 efC=40" "$logs\hnsw_m16_efc40_search.log" {
    & "$build\search_hnsw.exe" --index "$tmp\hnsw_m16_efc40.bin" --data $data --queries $queries --gt $gt --K 10 --ef "10,20,50,100,200,400"
}

# Search Vamana R=32 (existing baseline)
Run-Step "SEARCH Vamana R=32 Baseline" "$logs\vam_r32_baseline_search.log" {
    & "$build\search_index.exe" --index "$tmp\index_baseline.bin" --data $data --queries $queries --gt $gt --K 10 --L "10,20,50,75,100,200"
}

# Search Vamana R=64 (existing improved)
Run-Step "SEARCH Vamana R=64 ADC" "$logs\vam_r64_adc_search.log" {
    & "$build\search_index.exe" --index "$tmp\index_improved.bin" --data $data --queries $queries --gt $gt --K 10 --L "10,20,50,75,100,200" --quantized
}

Write-Host ""
Write-Host "===========================================================" -ForegroundColor Yellow
Write-Host "  BENCHMARK COMPLETE" -ForegroundColor Yellow
Write-Host "===========================================================" -ForegroundColor Yellow
