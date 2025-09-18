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

template <typename DType>
__global__ void normalize_vectors_kernel(int n_rows, int n_cols, DType* P) {
    // each block normalizes one row
    size_t row_idx = static_cast<size_t>(blockIdx.x);
    if (row_idx >= n_rows)
        return;

    // store partial sum of squares for each thread
    extern __shared__ DType sdata[];

    unsigned int tid = threadIdx.x;

    DType sum_sq = DType(0.0);
    for (size_t col_idx = tid; col_idx < n_cols; col_idx += blockDim.x) {
        DType val = P[row_idx * n_cols + col_idx];
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
        for (size_t col_idx = tid; col_idx < n_cols; col_idx += blockDim.x) {
            P[row_idx * n_cols + col_idx] *= inv_norm;
        }
    }
}

/**
 * @brief Sample n_rows random unit vectors from a n_cols-dimensional sphere.
 * @param[in] stream CUDA stream.
 * @param[in] n_rows Number of vectors to generate.
 * @param[in] n_cols Dimensionality of each vector.
 * @param[in] seed Seed for the random number generator.
 * @param[out] P n_rows x n_cols matrix of random unit vectors.
 */
template <typename DType>
void generate_random_projections(cudaStream_t stream, int n_rows, int n_cols, uint64_t seed,
                                 DType* P) {
    // create curand generator
    curandGenerator_t rng;
    CURAND_CHECK(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(rng, seed));

    size_t projection_size = static_cast<size_t>(n_rows) * n_cols;

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
    int block_size = std::min(n_cols, 1024);
    int smem_size = block_size * sizeof(DType);
    normalize_vectors_kernel<DType><<<n_rows, block_size, smem_size, stream>>>(n_rows, n_cols, P);

    CUDA_CHECK(cudaStreamSynchronize(stream)); // tbd - remove this sync?
    CURAND_CHECK(curandDestroyGenerator(rng));
}

} // namespace detail
} // namespace rplsh
} // namespace culsh
