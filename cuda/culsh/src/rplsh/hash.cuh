#pragma once

#include "../core/constants.cuh"
#include "../core/utils.cuh"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <device_launch_parameters.h>
#include <stdexcept>

namespace culsh {
namespace rplsh {
namespace detail {

/**
 * @brief Convert hashed input matrix to binary signatures, laid out contiguously by table
 * @param[in] X_hash Hashed input vectors (n_samples x n_hash_tables * n_hashes)
 * @param[in] n_samples Number of input vectors
 * @param[in] n_hash_tables Number of hash tables
 * @param[in] n_hashes Number of hashes per table
 * @param[out] X_sig Compressed output signatures (n_hash_tables x n_samples x n_hashes)
 */
template <typename DType>
__global__ void compute_signatures_kernel(const DType* X_hash, int n_samples, int n_hash_tables,
                                          int n_hashes, uint8_t* X_sig) {

    // each thread computes one signature
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total_elements = static_cast<size_t>(n_samples) * n_hash_tables * n_hashes;

    if (idx >= total_elements)
        return;

    size_t proj_idx = idx % n_hashes;
    size_t table_idx = (idx / n_hashes) % n_hash_tables;
    size_t row_idx = idx / (n_hashes * n_hash_tables);

    // Index of input hash
    size_t hash_idx = row_idx * (n_hash_tables * n_hashes) + table_idx * n_hashes + proj_idx;

    // Lay out signatures contiguously per table
    size_t sig_idx = table_idx * (n_samples * n_hashes) + row_idx * n_hashes + proj_idx;

    // Convert to binary
    // Reinterpret as uint32, shift to sign bit, and invert (1 is pos, 0 is neg)
    X_sig[sig_idx] = static_cast<uint8_t>((__float_as_uint(X_hash[hash_idx]) >> 31) ^ 1);
}

/**
 * @brief Sample n random vectors from a d-dimensional sphere
 * @param[in] stream CUDA stream
 * @param[in] n Number of vectors to generate
 * @param[in] d Dimensionality of each vector
 * @param[in] seed Seed for the random number generator
 * @param[out] P Device pointer to n x d array of random vectors (float or double)
 */
template <typename DType>
void generate_random_projections(cudaStream_t stream, int n, int d, uint64_t seed, DType* P) {
    curandGenerator_t rng;
    CURAND_CHECK(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetStream(rng, stream));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(rng, seed));

    size_t projection_size = static_cast<size_t>(n) * d;

    if constexpr (std::is_same_v<DType, float>) {
        // Generate random normal floats
        CURAND_CHECK(curandGenerateNormal(rng, P, projection_size, 0.0f, 1.0f));
    } else if constexpr (std::is_same_v<DType, double>) {
        // Generate random normal doubles
        CURAND_CHECK(curandGenerateNormalDouble(rng, P, projection_size, 0.0, 1.0));
    } else {
        throw std::invalid_argument("Expected float or double type");
    }

    CURAND_CHECK(curandDestroyGenerator(rng));
}

/**
 * @brief Hash input vectors X using projections P and compute binary signatures
 * @param[in] cublas_handle cuBLAS handle
 * @param[in] stream CUDA stream
 * @param[in] X Input vectors (n_samples x n_features)
 * @param[in] P Random projection matrix (n_hash_tables * n_hashes x n_features)
 * @param[in] n_samples Number of input vectors
 * @param[in] n_features Dimensionality of each input vector
 * @param[in] n_hash_tables Number of hash tables
 * @param[in] n_hashes Number of hashes per table
 * @param[out] X_sig Table-major binary signature matrix (n_hash_tables x n_samples x n_hashes)
 */
template <typename DType>
void hash(cublasHandle_t cublas_handle, cudaStream_t stream, const DType* X, const DType* P,
          int n_samples, int n_features, int n_hash_tables, int n_hashes, uint8_t* X_sig) {

    cublasSetStream(cublas_handle, stream);

    const DType alpha = DType(1.0);
    const DType beta = DType(0.0);
    const int n_total_buckets = n_hash_tables * n_hashes;

    // Allocate intermediate hash buffer
    size_t hash_size = static_cast<size_t>(n_samples) * n_total_buckets;
    DType* X_hash = nullptr;
    CUDA_CHECK(cudaMalloc(&X_hash, hash_size * sizeof(DType)));

    // Given row-major X, P, compute X * P^T = X_hash.
    // In col-major (used by cuBLAS - denoting with _c), we have X_c = X^T, P_c = P^T.
    // Thus to get X_hash in row-major, compute P_c^T * X_c = X_hash_c^T = X_hash.
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

    // Compute binary signatures and layout in table-major format
    dim3 block_size(core::BLOCK_SIZE);
    dim3 grid_size((hash_size + block_size.x - 1) / block_size.x);
    compute_signatures_kernel<<<grid_size, block_size, 0, stream>>>(X_hash, n_samples,
                                                                    n_hash_tables, n_hashes, X_sig);

    CUDA_CHECK(cudaFree(X_hash));
}

} // namespace detail
} // namespace rplsh
} // namespace culsh
