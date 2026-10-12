# SIFT1M results

**Setup**

* SIFT1M: 1,000,000 base vectors x 128 dimensions, 10,000 queries, official ground truth.
* Built with GCC 15.2, `-O3 -march=native`, on a 24-thread x86-64 laptop.
* Index builds use all 24 threads, `--single_pass`, build L=75, α=1.2, γ=1.5.
* Search is single-threaded (`OMP_NUM_THREADS=1`), so latency is per query with no contention between cores. QPS is per core (1e6 / mean latency).
* Latencies are medians of three runs; recall and distance-computation counts are deterministic.
* Absolute latencies depend on the machine; the ratios between configurations carry over (a 2-core cloud VM gave the same ordering, at about 2.5x the latency).
* Regenerate everything with `scripts/benchmark_sift1m.sh`.

| Index | Build flags | Average out-degree | Build time (24 threads) |
|---|---|---|---|
| Baseline R=32 | `--R 32 --baseline` | 33.1 | 35 s |
| Improved R=32 | `--R 32` | 38.8 | 47 s |
| Improved R=64 | `--R 64` | 75.8 | 134 s |

"Vectorised kernel" means the same binary built with `-DGRAPHANN_SIMD_REDUCTION=ON`. "uint8 ADC" means `search_index --quantized`, and "dynamic beam" adds `--dynamic`.

## Cheapest setting that reaches a recall target

Mean / P99 latency in µs.

| Recall@10 | Baseline R=32 | Improved R=32 | Improved R=64 | Improved R=64, vectorised |
|---|---|---|---|---|
| >= 0.98 | L=75: 301 / 440 | L=75: 286 / 378 | L=50: 349 / 459 (0.9916) | L=50: 275 / 363 |
| >= 0.99 | L=150: 528 / 760 | L=150: 496 / 670 | L=50: 349 / 459 | L=50: 275 / 363 |
| >= 0.995 | L=200: 661 / 973 | L=200: 633 / 882 | L=75: 469 / 631 | L=75: 376 / 503 |

## Full tables


### Baseline graph, R=32, exact float32

| L | Recall@10 | Distance comps | Mean µs | P99 µs | QPS/core |
|---|---|---|---|---|---|
| 10 | 0.7720 | 648 | 105 | 175 | 9,515 |
| 20 | 0.8876 | 876 | 136 | 218 | 7,348 |
| 30 | 0.9305 | 1,080 | 168 | 248 | 5,963 |
| 50 | 0.9648 | 1,464 | 228 | 367 | 4,388 |
| 75 | 0.9800 | 1,910 | 301 | 440 | 3,319 |
| 100 | 0.9870 | 2,330 | 375 | 525 | 2,669 |
| 150 | 0.9929 | 3,102 | 528 | 760 | 1,894 |
| 200 | 0.9952 | 3,818 | 661 | 973 | 1,513 |

### Baseline graph, R=32, vectorised kernel

| L | Recall@10 | Distance comps | Mean µs | P99 µs | QPS/core |
|---|---|---|---|---|---|
| 10 | 0.7720 | 648 | 78 | 127 | 12,804 |
| 20 | 0.8876 | 876 | 104 | 162 | 9,597 |
| 30 | 0.9305 | 1,080 | 129 | 194 | 7,740 |
| 50 | 0.9648 | 1,464 | 176 | 251 | 5,685 |
| 75 | 0.9800 | 1,910 | 237 | 333 | 4,216 |
| 100 | 0.9870 | 2,330 | 306 | 522 | 3,271 |
| 150 | 0.9929 | 3,102 | 425 | 740 | 2,355 |
| 200 | 0.9952 | 3,818 | 538 | 772 | 1,858 |

### Baseline graph, R=32, uint8 ADC

| L | Recall@10 | Distance comps | Mean µs | P99 µs | QPS/core |
|---|---|---|---|---|---|
| 10 | 0.7761 | 656 | 123 | 194 | 8,130 |
| 20 | 0.8923 | 896 | 177 | 278 | 5,659 |
| 30 | 0.9343 | 1,113 | 223 | 325 | 4,484 |
| 50 | 0.9676 | 1,523 | 313 | 441 | 3,192 |
| 75 | 0.9821 | 2,001 | 421 | 586 | 2,374 |
| 100 | 0.9886 | 2,450 | 533 | 764 | 1,876 |
| 150 | 0.9939 | 3,284 | 720 | 983 | 1,389 |
| 200 | 0.9961 | 4,060 | 922 | 1,294 | 1,084 |

### Improved graph, R=32, exact float32

| L | Recall@10 | Distance comps | Mean µs | P99 µs | QPS/core |
|---|---|---|---|---|---|
| 10 | 0.8010 | 550 | 86 | 125 | 11,614 |
| 20 | 0.8992 | 788 | 121 | 171 | 8,278 |
| 30 | 0.9377 | 1,009 | 153 | 226 | 6,532 |
| 50 | 0.9682 | 1,425 | 214 | 294 | 4,682 |
| 75 | 0.9823 | 1,906 | 286 | 378 | 3,495 |
| 100 | 0.9886 | 2,357 | 357 | 470 | 2,802 |
| 150 | 0.9937 | 3,186 | 496 | 670 | 2,015 |
| 200 | 0.9960 | 3,952 | 633 | 882 | 1,581 |

### Improved graph, R=32, vectorised kernel

| L | Recall@10 | Distance comps | Mean µs | P99 µs | QPS/core |
|---|---|---|---|---|---|
| 10 | 0.8010 | 550 | 68 | 100 | 14,793 |
| 20 | 0.8992 | 788 | 94 | 132 | 10,661 |
| 30 | 0.9377 | 1,009 | 118 | 162 | 8,439 |
| 50 | 0.9682 | 1,425 | 165 | 215 | 6,050 |
| 75 | 0.9823 | 1,906 | 224 | 327 | 4,454 |
| 100 | 0.9886 | 2,357 | 283 | 382 | 3,529 |
| 150 | 0.9937 | 3,186 | 398 | 573 | 2,513 |
| 200 | 0.9960 | 3,952 | 524 | 971 | 1,910 |

### Improved graph, R=64, exact float32

| L | Recall@10 | Distance comps | Mean µs | P99 µs | QPS/core |
|---|---|---|---|---|---|
| 10 | 0.8865 | 976 | 132 | 187 | 7,582 |
| 20 | 0.9567 | 1,422 | 192 | 258 | 5,208 |
| 30 | 0.9784 | 1,839 | 246 | 322 | 4,057 |
| 50 | 0.9916 | 2,608 | 349 | 459 | 2,867 |
| 75 | 0.9963 | 3,476 | 469 | 631 | 2,130 |
| 100 | 0.9978 | 4,272 | 588 | 829 | 1,701 |
| 150 | 0.9986 | 5,706 | 810 | 1,137 | 1,235 |
| 200 | 0.9991 | 7,004 | 1,025 | 1,467 | 976 |

### Improved graph, R=64, vectorised kernel

| L | Recall@10 | Distance comps | Mean µs | P99 µs | QPS/core |
|---|---|---|---|---|---|
| 10 | 0.8865 | 976 | 104 | 147 | 9,643 |
| 20 | 0.9567 | 1,422 | 151 | 202 | 6,640 |
| 30 | 0.9784 | 1,839 | 194 | 257 | 5,144 |
| 50 | 0.9916 | 2,608 | 275 | 363 | 3,634 |
| 75 | 0.9963 | 3,476 | 376 | 503 | 2,660 |
| 100 | 0.9978 | 4,272 | 473 | 660 | 2,114 |
| 150 | 0.9986 | 5,706 | 661 | 939 | 1,514 |
| 200 | 0.9991 | 7,004 | 846 | 1,318 | 1,181 |

### Improved graph, R=64, uint8 ADC

| L | Recall@10 | Distance comps | Mean µs | P99 µs | QPS/core |
|---|---|---|---|---|---|
| 10 | 0.8895 | 996 | 170 | 246 | 5,896 |
| 20 | 0.9614 | 1,468 | 261 | 384 | 3,834 |
| 30 | 0.9814 | 1,911 | 346 | 555 | 2,889 |
| 50 | 0.9931 | 2,730 | 494 | 670 | 2,026 |
| 75 | 0.9973 | 3,658 | 662 | 1,034 | 1,511 |
| 100 | 0.9983 | 4,510 | 816 | 1,090 | 1,225 |
| 150 | 0.9990 | 6,058 | 1,166 | 1,742 | 857 |
| 200 | 0.9993 | 7,461 | 1,436 | 2,002 | 696 |

### Improved graph, R=64, uint8 ADC + dynamic beam

| L | Recall@10 | Distance comps | Mean µs | P99 µs | QPS/core |
|---|---|---|---|---|---|
| 10 | 0.8895 | 996 | 171 | 239 | 5,841 |
| 20 | 0.9614 | 1,468 | 258 | 342 | 3,883 |
| 30 | 0.9814 | 1,911 | 343 | 495 | 2,917 |
| 50 | 0.9931 | 2,730 | 495 | 690 | 2,020 |
| 75 | 0.9973 | 3,658 | 653 | 893 | 1,532 |
| 100 | 0.9983 | 4,510 | 823 | 1,087 | 1,215 |
| 150 | 0.9990 | 6,058 | 1,145 | 1,556 | 874 |
| 200 | 0.9993 | 7,461 | 1,426 | 2,032 | 701 |
