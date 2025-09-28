#pragma once

#include "../utils/utils.cuh"
#include <cstdint>
#include <cuda_runtime.h>
#include <curand.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdexcept>

namespace culsh {
namespace rplsh {
namespace detail {

/**
 * @brief Calculate and apply row-wise normalization to matrix P.
 */
template <typename DType>
__global__ void normalize_vectors_kernel(int n_samples, int n_features, DType* P) {
    // each block normalizes one row
    size_t row_idx = static_cast<size_t>(blockIdx.x);
    if (row_idx >= n_samples)
        return;

    // store partial sum of squares for each thread
    extern __shared__ char sdata_raw[];
    DType* sdata =
        reinterpret_cast<DType*>(sdata_raw); // can't directly initialize with template type

    unsigned int tid = threadIdx.x;

    DType sum_sq = DType(0.0);
    for (size_t col_idx = tid; col_idx < n_features; col_idx += blockDim.x) {
        DType val = P[row_idx * n_features + col_idx];
        sum_sq += val * val;
    }
    sdata[tid] = sum_sq;
    __syncthreads();

    // reduce to get sum of squares for row
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    DType row_norm = sqrt(sdata[0]);

    // normalize row
    if (row_norm > DType(1e-8)) { // check div by zero
        DType inv_norm = DType(1.0) / row_norm;
        for (size_t col_idx = tid; col_idx < n_features; col_idx += blockDim.x) {
            P[row_idx * n_features + col_idx] *= inv_norm;
        }
    }
}

/**
 * @brief Sample n_samples random unit vectors from a n_features-dimensional sphere
 * @param[in] stream CUDA stream
 * @param[in] n_samples Number of vectors to generate
 * @param[in] n_features Dimensionality of each vector
 * @param[in] seed Seed for the random number generator
 * @param[out] P n_samples x n_features matrix of random unit vectors
 */
template <typename DType>
void generate_random_projections(cudaStream_t stream, int n_samples, int n_features, uint64_t seed,
                                 DType* P) {
    // create curand generator
    curandGenerator_t rng;
    CURAND_CHECK(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(rng, seed));

    size_t projection_size = static_cast<size_t>(n_samples) * n_features;

    if constexpr (std::is_same_v<DType, float>) {
        // generate random normal floats
        CURAND_CHECK(curandGenerateNormal(rng, P, projection_size, 0.0f, 1.0f));
    } else if constexpr (std::is_same_v<DType, double>) {
        // generate random normal doubles
        CURAND_CHECK(curandGenerateNormalDouble(rng, P, projection_size, 0.0, 1.0));
    } else {
        throw std::invalid_argument("Expected float or double type");
    }

    // launch row-wise norm kernel
    int block_size = std::min(n_features, 256);
    int smem_size = block_size * sizeof(DType);
    normalize_vectors_kernel<DType>
        <<<n_samples, block_size, smem_size, stream>>>(n_samples, n_features, P);

    CURAND_CHECK(curandDestroyGenerator(rng));
}

} // namespace detail
} // namespace rplsh
} // namespace culsh
