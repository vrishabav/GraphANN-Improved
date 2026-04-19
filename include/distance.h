#pragma once

#include <cstdint>
#include <cfloat>

// Squared L2 (Euclidean) distance between two float vectors.
// No sqrt — monotonic, so rankings are preserved.
// Compiler auto-vectorizes this with -O3 -march=native.
float compute_l2sq(const float *a, const float *b, uint32_t dim);

// Early-abandoning L2 squared: aborts if partial sum exceeds threshold.
// Checks every 32 dimensions to preserve SIMD auto-vectorization.
// Returns FLT_MAX on early abandon.
float compute_l2sq_ea(const float *a, const float *b, uint32_t dim,
                      float threshold);

// Asymmetric L2 squared: float32 query vs uint8 quantized vector.
// Dequantizes each dimension on-the-fly: reconstructed = quantized[d] * scale[d] + min[d]
// Used for fast graph traversal; final top-K are re-ranked with exact float32.
float compute_l2sq_asymmetric(const float *query, const uint8_t *quantized,
                              const float *dim_min, const float *dim_scale,
                              uint32_t dim);

// Early-abandoning asymmetric L2 squared (float32 query vs uint8 quantized).
// Checks every 32 dimensions; returns FLT_MAX on early abandon.
float compute_l2sq_asymmetric_ea(const float *query, const uint8_t *quantized,
                                 const float *dim_min, const float *dim_scale,
                                 uint32_t dim, float threshold);

// L2 squared distance in PCA-projected space (lower dimensional).
// proj_query: pre-projected query vector (pca_dim floats)
// proj_data:  pre-projected dataset vector (pca_dim floats)
float compute_l2sq_pca(const float *proj_query, const float *proj_data,
                       uint32_t pca_dim);