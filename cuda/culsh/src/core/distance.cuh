#pragma once

namespace culsh {
namespace core {

/**
 * @brief L2 distance functor
 */
struct L2Distance {
    __device__ __forceinline__ 
    float operator()(const float* a, const float* b, int d) const {
        float sum = 0.0f;
        for (int i = 0; i < d; i++) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum; // skip sqrt for ranking
    }
};

/**
 * @brief Cosine distance functor
 */
struct CosineDistance {
    __device__ __forceinline__
    float operator()(const float* a, const float* b, int d) const {
        float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
        for (int i = 0; i < d; i++) {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        float denom = sqrtf(norm_a) * sqrtf(norm_b) + 1e-8f;
        return 1.0f - (dot / denom);
    }
};

} // namespace core
} // namespace culsh
