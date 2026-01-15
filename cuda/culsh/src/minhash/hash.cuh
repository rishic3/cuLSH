#pragma once

#include "../core/constants.cuh"
#include "../core/utils.cuh"
#include <cub/block/block_reduce.cuh>
#include <cuda_runtime.h>
#include <curand.h>
#include <device_launch_parameters.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>

namespace culsh {
namespace minhash {
namespace detail {

static constexpr int HASH_TILE = 16;
static constexpr uint32_t HASH_PRIME = 4294967291u; // Largest 32-bit prime

/**
 * @brief Normalize random integers to ensure A is non-zero and A, B < PRIME
 * @param[in] stream CUDA stream
 * @param[in] n_hashes Number of hash functions to generate
 * @param[in] A Device pointer to array of n random integers
 * @param[in] B Device pointer to array of n random integers
 */
void normalize_hash_integers(cudaStream_t stream, int n_hashes, uint32_t* A, uint32_t* B) {
    thrust::transform(thrust::cuda::par.on(stream), A, A + n_hashes, A,
                      (thrust::placeholders::_1 % (HASH_PRIME - 1)) + 1);

    thrust::transform(thrust::cuda::par.on(stream), B, B + n_hashes, B,
                      thrust::placeholders::_1 % HASH_PRIME);
}

/**
 * @brief Generate random integers to construct 2-universal hash functions
 * @param[in] stream CUDA stream
 * @param[in] n Number of hash functions to generate
 * @param[in] seed Seed for the random number generator
 * @param[out] A Device pointer to array of n random integers
 * @param[out] B Device pointer to array of n random integers
 */
void generate_hash_integers(cudaStream_t stream, int n, uint64_t seed, uint32_t* A, uint32_t* B) {
    curandGenerator_t rng;
    CURAND_CHECK(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetStream(rng, stream));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(rng, seed));

    CURAND_CHECK(curandGenerate(rng, A, n));
    CURAND_CHECK(curandGenerate(rng, B, n));

    normalize_hash_integers(stream, n, A, B);

    CURAND_CHECK(curandDestroyGenerator(rng));
}

/**
 * @brief Compute minhash of each input row
 * @param[in] X_indices Indices for input CSR matrix (nnz)
 * @param[in] X_indptr Indptr for input CSR matrix (n_samples + 1)
 * @param[in] A Device pointer to array of n random integers
 * @param[in] B Device pointer to array of n random integers
 * @param[in] p Prime integer to construct hash functions
 * @param[in] n_samples Number of input samples
 * @param[in] n_features Dimensionality of each input sample
 * @param[in] n_hash_tables Number of hash tables
 * @param[in] n_hashes Number of hashes per table
 * @param[out] X_sig Table-major minhash signatures (n_hash_tables x n_samples x n_hashes)
 */
__global__ void compute_minhash_kernel(const int* __restrict__ X_indices,
                                       const int* __restrict__ X_indptr,
                                       const uint32_t* __restrict__ A,
                                       const uint32_t* __restrict__ B, int n_samples,
                                       int n_hash_tables, int n_hashes,
                                       uint32_t* __restrict__ X_sig) {
    int total_hashes = n_hashes * n_hash_tables;
    // Kernel is launched in 2D block of (row, hash fn)
    // Each block handles 1 row (1 thread per element) and HASH_TILE hash functions
    int row = blockIdx.x;
    int tile = blockIdx.y;
    int h_0 = tile * HASH_TILE;
    if (row >= n_samples)
        return;

    // Get start and end index of row into X_indices
    int start = X_indptr[row];
    int end = X_indptr[row + 1];

    // Initialize local minhash array to uint32_t maxval
    uint32_t minima[HASH_TILE];
    for (int t = 0; t < HASH_TILE; ++t) {
        minima[t] = 0xFFFFFFFFu;
    }

    // Each loop iteration computes BLOCK_SIZE elements of the row
    for (int j = start + threadIdx.x; j < end; j += blockDim.x) {
        // Get this thread's row value
        uint32_t x_ij = static_cast<uint32_t>(X_indices[j]);

        // Apply each hash function and update local min
        for (int t = 0; t < HASH_TILE; ++t) {
            int h = h_0 + t;
            if (h < total_hashes) {
                uint64_t tmp = (uint64_t)A[h] * x_ij + (uint64_t)B[h];
                uint32_t hashed = uint32_t(tmp % HASH_PRIME);
                minima[t] = min(minima[t], hashed);
            }
        }
    }

    // Reduce to get block-wise minimum for each hash fn
    using BlockReduce = cub::BlockReduce<uint32_t, core::BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage tmp;
    for (int t = 0; t < HASH_TILE; ++t) {
        uint32_t block_min = BlockReduce(tmp).Reduce(minima[t], cub::Min());
        __syncthreads();

        // First thread writes minhash to signature
        int h = h_0 + t;
        if (threadIdx.x == 0 && h < total_hashes) {
            // Write table-major X_sig[table][row][hash_within_table]
            int table_id = h / n_hashes;
            int hash_id = h % n_hashes;
            size_t table_stride = static_cast<size_t>(n_samples) * n_hashes;
            X_sig[static_cast<size_t>(table_id) * table_stride +
                  static_cast<size_t>(row) * n_hashes + hash_id] = block_min;
        }
    }
}

/**
 * @brief Compute minhash signatures from input samples
 * @param[in] stream CUDA stream
 * @param[in] X_indices Indices for input CSR matrix (nnz)
 * @param[in] X_indptr Indptr for input CSR matrix (n_samples + 1)
 * @param[in] A Device pointer to array of n random integers
 * @param[in] B Device pointer to array of n random integers
 * @param[in] n_samples Number of input samples
 * @param[in] n_hash_tables Number of hash tables
 * @param[in] n_hashes Number of hashes per table
 * @param[out] X_sig Table-major minhash signatures (n_hash_tables x n_samples x n_hashes)
 */
void compute_minhash(cudaStream_t stream, const int* X_indices, const int* X_indptr,
                     const uint32_t* A, const uint32_t* B, int n_samples, int n_hash_tables,
                     int n_hashes, uint32_t* X_sig) {
    int total_hashes = n_hash_tables * n_hashes;
    int n_tiles = (total_hashes + HASH_TILE - 1) / HASH_TILE;
    dim3 grid_size(n_samples, n_tiles);
    dim3 block_size(core::BLOCK_SIZE);

    // For each input row X_i and each hash h_k, compute sig[i,k] = min(h_i(X_ij)) over input
    // columns j. Launch one block per (row, hash tile). Each block is responsible for the minhash
    // of one row under HASH_TILE hash functions.
    compute_minhash_kernel<<<grid_size, block_size, 0, stream>>>(
        X_indices, X_indptr, A, B, n_samples, n_hash_tables, n_hashes, X_sig);
}

} // namespace detail
} // namespace minhash
} // namespace culsh
