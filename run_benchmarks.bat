@echo off
setlocal

set BASE=%~dp0
set BUILD=%BASE%\build
set TMP=%BASE%\tmp
set DATA=%TMP%\sift_base.fbin
set QUERIES=%TMP%\sift_query.fbin
set GT=%TMP%\sift_gt.ibin

echo ============================================================
echo HNSW BENCHMARK SUITE
echo ============================================================

REM ---- HNSW M=16 efC=100 ----
echo.
echo [1/5] Building HNSW M=16 efC=100...
%BUILD%\build_hnsw.exe --data %DATA% --output %TMP%\hnsw_m16_efc100.bin --M 16 --efC 100
if errorlevel 1 goto error

echo.
echo [1/5] Searching HNSW M=16 efC=100...
%BUILD%\search_hnsw.exe --index %TMP%\hnsw_m16_efc100.bin --data %DATA% --queries %QUERIES% --gt %GT% --K 10 --ef 10,20,50,100,200,400
if errorlevel 1 goto error

REM ---- HNSW M=16 efC=200 ----
echo.
echo [2/5] Building HNSW M=16 efC=200...
%BUILD%\build_hnsw.exe --data %DATA% --output %TMP%\hnsw_m16_efc200.bin --M 16 --efC 200
if errorlevel 1 goto error

echo.
echo [2/5] Searching HNSW M=16 efC=200...
%BUILD%\search_hnsw.exe --index %TMP%\hnsw_m16_efc200.bin --data %DATA% --queries %QUERIES% --gt %GT% --K 10 --ef 10,20,50,100,200,400
if errorlevel 1 goto error

REM ---- HNSW M=32 efC=200 ----
echo.
echo [3/5] Building HNSW M=32 efC=200...
%BUILD%\build_hnsw.exe --data %DATA% --output %TMP%\hnsw_m32_efc200.bin --M 32 --efC 200
if errorlevel 1 goto error

echo.
echo [3/5] Searching HNSW M=32 efC=200...
%BUILD%\search_hnsw.exe --index %TMP%\hnsw_m32_efc200.bin --data %DATA% --queries %QUERIES% --gt %GT% --K 10 --ef 10,20,50,100,200,400
if errorlevel 1 goto error

REM ---- Vamana baseline (existing index_improved.bin = R=64 best config) ----
echo.
echo [4/5] Searching Vamana best config (R=64, existing index)...
%BUILD%\search_index.exe --index %TMP%\index_improved.bin --data %DATA% --queries %QUERIES% --gt %GT% --K 10 --L 10,20,50,75,100,200 --quantized
if errorlevel 1 goto error

REM ---- Soft prune Vamana (R=32, single pass for speed) ----
echo.
echo [5/5] Building Vamana with soft diversity pruning (R=32, single pass)...
%BUILD%\build_index.exe --data %DATA% --output %TMP%\index_soft_prune.bin --R 32 --L 75 --alpha 1.2 --gamma 1.5 --single_pass --soft_prune --soft_beta 0.5
if errorlevel 1 goto error

echo.
echo [5/5] Searching Vamana soft prune...
%BUILD%\search_index.exe --index %TMP%\index_soft_prune.bin --data %DATA% --queries %QUERIES% --gt %GT% --K 10 --L 10,20,50,75,100,200
if errorlevel 1 goto error

REM ---- Vamana standard (R=32, single pass for comparison) ----
echo.
echo [5b] Building Vamana standard (R=32, single pass, for comparison)...
%BUILD%\build_index.exe --data %DATA% --output %TMP%\index_standard_r32.bin --R 32 --L 75 --alpha 1.2 --gamma 1.5 --single_pass
if errorlevel 1 goto error

echo.
echo [5b] Searching Vamana standard R=32...
%BUILD%\search_index.exe --index %TMP%\index_standard_r32.bin --data %DATA% --queries %QUERIES% --gt %GT% --K 10 --L 10,20,50,75,100,200
if errorlevel 1 goto error

echo.
echo ============================================================
echo ALL BENCHMARKS COMPLETE
echo ============================================================
goto end

:error
echo ERROR: benchmark step failed
exit /b 1

:end
endlocal
