#include "distance.h"
#ifdef _WIN32
#include <malloc.h>
#define aligned_alloc(alignment, size) _aligned_malloc(size, alignment)
#define aligned_free(ptr) _aligned_free(ptr)
#else
#include <cstdlib>
#define aligned_free(ptr) free(ptr)
#endif

float compute_l2sq(const float *a, const float *b, uint32_t dim)
{
    float sum = 0.0f;
#ifdef GRAPHANN_SIMD_REDUCTION
    // Without -ffast-math the compiler must add the 128 squared differences
    // strictly in order, which turns the reduction into one long serial chain
    // of scalar adds even though the subtract/multiply are vectorised. Allowing
    // a reordered (tree) reduction lets it keep 8-wide accumulators.
#pragma omp simd reduction(+ : sum)
#endif
    for (uint32_t i = 0; i < dim; i++)
    {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

float compute_l2sq_ea(const float *a, const float *b, uint32_t dim,
                      float threshold)
{
    float sum = 0.0f;
    uint32_t i = 0;
    // Process in 16-dim blocks — inner loop auto-vectorizes with -O3
    for (; i + 16 <= dim; i += 16)
    {
#ifdef GRAPHANN_SIMD_REDUCTION
        float block = 0.0f;
#pragma omp simd reduction(+ : block)
        for (uint32_t j = i; j < i + 16; j++)
        {
            float diff = a[j] - b[j];
            block += diff * diff;
        }
        sum += block;
#else
        for (uint32_t j = i; j < i + 16; j++)
        {
            float diff = a[j] - b[j];
            sum += diff * diff;
        }
#endif
        if (sum > threshold)
            return FLT_MAX; // early abandon
    }
    // Remaining dimensions (dim % 16)
    for (; i < dim; i++)
    {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

float compute_l2sq_asymmetric(const float *query, const uint8_t *quantized,
                              const float *dim_min, const float *dim_scale,
                              uint32_t dim)
{
    float sum = 0.0f;
    for (uint32_t i = 0; i < dim; i++)
    {
        float reconstructed = quantized[i] * dim_scale[i] + dim_min[i];
        float diff = query[i] - reconstructed;
        sum += diff * diff;
    }
    return sum;
}

float compute_l2sq_asymmetric_ea(const float *query, const uint8_t *quantized,
                                 const float *dim_min, const float *dim_scale,
                                 uint32_t dim, float threshold)
{
    float sum = 0.0f;
    uint32_t i = 0;
    // Process in 16-dim blocks — inner loop auto-vectorizes with -O3
    for (; i + 16 <= dim; i += 16)
    {
        for (uint32_t j = i; j < i + 16; j++)
        {
            float reconstructed = quantized[j] * dim_scale[j] + dim_min[j];
            float diff = query[j] - reconstructed;
            sum += diff * diff;
        }
        if (sum > threshold)
            return FLT_MAX; // early abandon
    }
    // Remaining dimensions (dim % 16)
    for (; i < dim; i++)
    {
        float reconstructed = quantized[i] * dim_scale[i] + dim_min[i];
        float diff = query[i] - reconstructed;
        sum += diff * diff;
    }
    return sum;
}

float compute_l2sq_pca(const float *proj_query, const float *proj_data,
                       uint32_t pca_dim)
{
    float sum = 0.0f;
    for (uint32_t i = 0; i < pca_dim; i++)
    {
        float diff = proj_query[i] - proj_data[i];
        sum += diff * diff;
    }
    return sum;
}