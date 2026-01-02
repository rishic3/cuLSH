#pragma once

#include "../core/constants.cuh"
#include "../core/utils.cuh"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdexcept>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>

namespace culsh {
namespace pslsh {
namespace detail {

/**
 * @brief Compute bias-add and floor to X_hash and layout table-major
 * @param[in] X_hash Hashed input vectors (n_hash_tables x n_samples x n_hashes)
 * @param[in] b Bias terms (n_hash_tables * n_hashes)
 * @param[in] n_samples Number of input vectors
 * @param[in] n_hash_tables Number of hash tables
 * @param[in] n_hashes Number of hashes per table
 * @param[out] X_sig Table-major signature matrix (n_hash_tables x n_samples x n_hashes)
 */
template <typename DType>
__global__ void bias_floor_reorder_kernel(const DType* X_hash, const DType* b, int n_samples,
                                          int n_hash_tables, int n_hashes, int32_t* X_sig) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total_elements = static_cast<size_t>(n_samples) * n_hash_tables * n_hashes;
    if (idx >= total_elements) {
        return;
    }

    size_t proj_idx = idx % n_hashes;
    size_t table_idx = (idx / n_hashes) % n_hash_tables;
    size_t row_idx = idx / (n_hashes * n_hash_tables);

    // Index of input hash (row-major)
    size_t hash_idx = row_idx * (n_hash_tables * n_hashes) + table_idx * n_hashes + proj_idx;
    // Index of output hash (table-major)
    size_t sig_idx = table_idx * (n_samples * n_hashes) + row_idx * n_hashes + proj_idx;
    // Index of bias term for this hash
    int bias_idx = table_idx * n_hashes + proj_idx;

    // Apply bias, floor, and write
    X_sig[sig_idx] = static_cast<int32_t>(floor(X_hash[hash_idx] + b[bias_idx]));
}

/**
 * @brief Generate n d-dimensional projections and n bias terms, normalized by window size
 * @param[in] stream CUDA stream
 * @param[in] n Number of vectors to generate
 * @param[in] d Dimensionality of each vector
 * @param[in] w Window size
 * @param[in] seed Seed for the random number generator
 * @param[out] P Device pointer to n x d array of random vectors (float or double)
 * @param[out] b Device pointer to n array of random bias terms
 */
template <typename DType>
void generate_random_projections_biases(cudaStream_t stream, int n, int d, int w, uint64_t seed,
                                        DType* P, DType* b) {
    curandGenerator_t rng;
    CURAND_CHECK(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetStream(rng, stream));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(rng, seed));

    size_t projection_size = static_cast<size_t>(n) * d;

    if constexpr (std::is_same_v<DType, float>) {
        // Generate random projections
        CURAND_CHECK(curandGenerateNormal(rng, P, projection_size, 0.0f, 1.0f));
        // Generate uniform bias terms in (0, 1]
        // Bias terms should be drawn from (0, w], and normalized by w; (0, 1] is already correct
        CURAND_CHECK(curandGenerateUniform(rng, b, n));
    } else if constexpr (std::is_same_v<DType, double>) {
        // Generate random projections
        CURAND_CHECK(curandGenerateNormalDouble(rng, P, projection_size, 0.0, 1.0));
        // Generate uniform bias terms in (0, 1]
        CURAND_CHECK(curandGenerateUniformDouble(rng, b, n));
    } else {
        throw std::invalid_argument("Expected float or double type");
    }

    // Normalize projections by window size
    thrust::transform(thrust::cuda::par.on(stream), P, P + projection_size, P,
                      thrust::placeholders::_1 / w);

    CURAND_CHECK(curandDestroyGenerator(rng));
}

/**
 * @brief Hash the input vectors X using projections P and bias terms b
 * @param[in] cublas_handle cuBLAS handle
 * @param[in] stream CUDA stream
 * @param[in] X Input vectors (n_samples x n_features)
 * @param[in] P Normalized projection matrix (n_hash_tables * n_hashes x n_features)
 * @param[in] b Normalized bias terms (n_hash_tables * n_hashes)
 * @param[in] n_samples Number of input vectors
 * @param[in] n_features Dimensionality of each input vector
 * @param[in] n_hash_tables Number of hash tables
 * @param[in] n_hashes Number of hashes per table
 * @param[out] X_sig Table-major signature matrix (n_hash_tables x n_samples x n_hashes)
 */
template <typename DType>
void hash(cublasHandle_t cublas_handle, cudaStream_t stream, const DType* X, const DType* P,
          const DType* b, int n_samples, int n_features, int n_hash_tables, int n_hashes,
          int32_t* X_sig) {

    cublasSetStream(cublas_handle, stream);

    const DType alpha = DType(1.0);
    const DType beta = DType(0.0);
    const int n_total_buckets = n_hash_tables * n_hashes;

    // Allocate intermediate hash buffer
    size_t hash_size = static_cast<size_t>(n_samples) * n_total_buckets;
    DType* X_hash = nullptr;
    CUDA_CHECK(cudaMalloc(&X_hash, hash_size * sizeof(DType)));

    // hash(x) = floor((P * x + b) / W) for each sample x in X.
    // Input P and bias b are assumed to be pre-normalized by window size W.
    // Compute X * P^T via cuBLAS GEMM.
    if constexpr (std::is_same_v<DType, float>) {
        CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, n_total_buckets,
                                 n_samples, n_features, &alpha, P, n_features, X, n_features, &beta,
                                 X_hash, n_total_buckets));
    } else if constexpr (std::is_same_v<DType, double>) {
        CUBLAS_CHECK(cublasDgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, n_total_buckets,
                                 n_samples, n_features, &alpha, P, n_features, X, n_features, &beta,
                                 X_hash, n_total_buckets));
    } else {
        throw std::invalid_argument("Expected float or double dtype");
    }

    // Compute floor(X_hash + b) and write to table-major X_sig
    dim3 block_size(core::BLOCK_SIZE);
    dim3 grid_size((hash_size + block_size.x - 1) / block_size.x);
    bias_floor_reorder_kernel<<<grid_size, block_size, 0, stream>>>(X_hash, b, n_samples,
                                                                    n_hash_tables, n_hashes, X_sig);

    CUDA_CHECK(cudaFree(X_hash));
}

} // namespace detail
} // namespace pslsh
} // namespace culsh
