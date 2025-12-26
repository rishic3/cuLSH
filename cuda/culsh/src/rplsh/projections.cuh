#pragma once

#include "../core/constants.cuh"
#include "../core/utils.cuh"
#include <cstdint>
#include <cub/block/block_reduce.cuh>
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
 * @param[in] n_samples Number of samples
 * @param[in] n_features Number of features
 * @param[in] P Device pointer to n_samples x n_features array of random unit vectors
 */
template <typename DType>
__global__ void normalize_vectors_kernel(int n_samples, int n_features, DType* P) {
    // Each block normalizes one row
    size_t row_idx = static_cast<size_t>(blockIdx.x);
    if (row_idx >= n_samples)
        return;

    // Compute partial sum of squares for this row
    unsigned int tid = threadIdx.x;
    DType sum_sq = DType(0.0);
    for (size_t col_idx = tid; col_idx < static_cast<size_t>(n_features); col_idx += blockDim.x) {
        DType val = P[row_idx * static_cast<size_t>(n_features) + col_idx];
        sum_sq += val * val;
    }

    // Block-wide reduce to get sum of squares for row
    using BlockReduce = cub::BlockReduce<DType, core::BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ DType row_norm;

    DType sum_sq_total = BlockReduce(temp_storage).Sum(sum_sq);
    if (tid == 0) {
        row_norm = sqrt(sum_sq_total);
    }
    __syncthreads();

    // Normalize row
    if (row_norm > DType(1e-8)) { // check div by zero
        DType inv_norm = DType(1.0) / row_norm;
        for (size_t col_idx = tid; col_idx < static_cast<size_t>(n_features);
             col_idx += blockDim.x) {
            P[row_idx * static_cast<size_t>(n_features) + col_idx] *= inv_norm;
        }
    }
}

/**
 * @brief Sample n_samples random unit vectors from an n_features-dimensional sphere
 * @param[in] stream CUDA stream
 * @param[in] n_samples Number of vectors to generate
 * @param[in] n_features Dimensionality of each vector
 * @param[in] seed Seed for the random number generator
 * @param[out] P Device pointer to n_samples x n_features array of random unit vectors
 */
template <typename DType>
void generate_random_projections(cudaStream_t stream, int n_samples, int n_features, uint64_t seed,
                                 DType* P) {
    curandGenerator_t rng;
    CURAND_CHECK(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetStream(rng, stream));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(rng, seed));

    size_t projection_size = static_cast<size_t>(n_samples) * n_features;

    if constexpr (std::is_same_v<DType, float>) {
        // Generate random normal floats
        CURAND_CHECK(curandGenerateNormal(rng, P, projection_size, 0.0f, 1.0f));
    } else if constexpr (std::is_same_v<DType, double>) {
        // Generate random normal doubles
        CURAND_CHECK(curandGenerateNormalDouble(rng, P, projection_size, 0.0, 1.0));
    } else {
        throw std::invalid_argument("Expected float or double type");
    }

    // Launch row-wise norm kernel
    normalize_vectors_kernel<DType>
        <<<n_samples, core::BLOCK_SIZE, 0, stream>>>(n_samples, n_features, P);

    CURAND_CHECK(curandDestroyGenerator(rng));
}

} // namespace detail
} // namespace rplsh
} // namespace culsh
