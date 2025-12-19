#pragma once

#include "../index.cuh"
#include "../utils/utils.cuh"
#include <cmath>
#include <cstdint>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/unique.h>

namespace culsh {
namespace rplsh {
namespace detail {

/**
 * @brief Pack hash values into sortable keys.
 *
 * Keys are stored as (table_id << n_projections) | packed_signature
 */
__global__ void pack_keys_kernel(const float* X_hash, int n_samples, int n_hash_tables,
                                 int n_projections, size_t n_items, uint32_t* keys) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n_items)
        return;

    // idx encodes (table_id * n_samples + sample_id)
    int table_id = static_cast<int>(idx / n_samples);
    int sample_id = static_cast<int>(idx % n_samples);

    // pack signature bits from hash values
    // X_hash layout: [sample][table][projection]
    uint32_t sig = 0;
    for (int p = 0; p < n_projections; ++p) {
        size_t hash_idx = static_cast<size_t>(sample_id) * n_hash_tables * n_projections +
                          static_cast<size_t>(table_id) * n_projections + p;
        // set sign bit
        uint32_t bit = (X_hash[hash_idx] >= 0.0f) ? 1u : 0u;
        sig = (sig << 1) | bit;
    }

    // combine table_id (high bits) with signature (low bits)
    keys[idx] = (static_cast<uint32_t>(table_id) << n_projections) | sig;
}

/**
 * @brief Detect bucket boundaries.
 */
__global__ void detect_boundaries_kernel(const uint32_t* keys, size_t n_items, int* bucket_flags) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n_items)
        return;

    if (idx == 0) {
        bucket_flags[idx] = 1;
    } else {
        bucket_flags[idx] = (keys[idx] != keys[idx - 1]) ? 1 : 0;
    }
}

/**
 * @brief Extract original sample IDs from sorted item indices.
 */
__global__ void extract_sample_ids_kernel(const uint32_t* item_indices, int n_samples,
                                          size_t n_items, int* candidate_indices) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n_items)
        return;

    candidate_indices[idx] = static_cast<int>(item_indices[idx] % n_samples);
}

/**
 * @brief Compute table start key for binary search.
 */
struct TableStartKeyFunctor {
    int n_projections;

    __host__ __device__ TableStartKeyFunctor(int n_projections) : n_projections(n_projections) {}

    __device__ uint32_t operator()(int table_id) const {
        return static_cast<uint32_t>(table_id) << n_projections;
    }
};

/**
 * @brief Check if flag equals 1.
 */
struct IsOneFunctor {
    __host__ __device__ bool operator()(int flag) const { return flag == 1; }
};

/**
 * @brief Build flat LSH index.
 *
 * @param[in] stream CUDA stream
 * @param[in] X_hash Hash values (n_samples x n_hash_tables * n_projections)
 * @param[in] n_samples Number of input samples
 * @param[in] n_hash_tables Number of hash tables
 * @param[in] n_projections Number of projections per table
 * @return Index Device index structure
 */
Index build_index(cudaStream_t stream, const float* X_hash, int n_samples, int n_hash_tables,
                  int n_projections) {

    // Validate that we can fit table_id + signature in uint32_t
    // Key format: (table_id << n_projections) | signature
    // table_id needs ceil(log2(n_hash_tables)) bits, signature needs n_projections bits
    int table_id_bits = static_cast<int>(std::ceil(std::log2(n_hash_tables)));
    int total_bits = table_id_bits + n_projections;
    if (total_bits > 32) {
        fprintf(
            stderr,
            "Error: n_hash_tables=%d (%d bits) + n_projections=%d exceeds 32-bit key capacity\n",
            n_hash_tables, table_id_bits, n_projections);
        exit(1);
    }

    auto policy = thrust::cuda::par.on(stream);
    size_t n_items = static_cast<size_t>(n_samples) * n_hash_tables;

    dim3 block_size(256);
    dim3 grid_size((n_items + block_size.x - 1) / block_size.x);

    // Pack (table_id, signature) into sortable keys
    uint32_t* d_keys;
    uint32_t* d_keys_sorted;
    uint32_t* d_item_indices;
    uint32_t* d_item_indices_sorted;

    CUDA_CHECK(cudaMalloc(&d_keys, n_items * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_keys_sorted, n_items * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_item_indices, n_items * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_item_indices_sorted, n_items * sizeof(uint32_t)));

    thrust::sequence(policy, d_item_indices, d_item_indices + n_items, 0u);
    pack_keys_kernel<<<grid_size, block_size, 0, stream>>>(X_hash, n_samples, n_hash_tables,
                                                           n_projections, n_items, d_keys);

    // Radix sort by packed key
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_keys_sorted,
                                    d_item_indices, d_item_indices_sorted, n_items, 0,
                                    sizeof(uint32_t) * 8, stream);
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_keys_sorted,
                                    d_item_indices, d_item_indices_sorted, n_items, 0,
                                    sizeof(uint32_t) * 8, stream);

    CUDA_CHECK(cudaFree(d_keys));
    CUDA_CHECK(cudaFree(d_item_indices));

    // Detect bucket boundaries
    int* d_bucket_flags;
    CUDA_CHECK(cudaMalloc(&d_bucket_flags, n_items * sizeof(int)));

    detect_boundaries_kernel<<<grid_size, block_size, 0, stream>>>(d_keys_sorted, n_items,
                                                                   d_bucket_flags);

    // Compute bucket IDs
    int* d_bucket_ids;
    CUDA_CHECK(cudaMalloc(&d_bucket_ids, n_items * sizeof(int)));

    thrust::exclusive_scan(policy, d_bucket_flags, d_bucket_flags + n_items, d_bucket_ids, 0);

    // Get total number of buckets
    int last_bucket_id, last_flag;
    CUDA_CHECK(cudaMemcpyAsync(&last_bucket_id, d_bucket_ids + (n_items - 1), sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&last_flag, d_bucket_flags + (n_items - 1), sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int n_total_buckets = last_bucket_id + last_flag;

    // Extract unique bucket keys
    Index index;
    index.n_hash_tables = n_hash_tables;
    index.n_projections = n_projections;
    index.n_total_buckets = n_total_buckets;
    index.n_total_candidates = static_cast<int>(n_items);

    CUDA_CHECK(cudaMalloc(&index.bucket_keys, n_total_buckets * sizeof(uint32_t)));

    thrust::unique_copy(policy, d_keys_sorted, d_keys_sorted + n_items, index.bucket_keys);

    // Extract bucket candidate offsets
    CUDA_CHECK(cudaMalloc(&index.bucket_candidate_offsets, (n_total_buckets + 1) * sizeof(int)));

    // Copy positions where bucket_flags == 1
    auto flag_iter = thrust::make_transform_iterator(d_bucket_flags, IsOneFunctor());

    thrust::copy_if(policy, thrust::make_counting_iterator<int>(0),
                    thrust::make_counting_iterator<int>(static_cast<int>(n_items)), flag_iter,
                    index.bucket_candidate_offsets, thrust::identity<bool>());

    // Set sentinel value
    int n_items_int = static_cast<int>(n_items);
    CUDA_CHECK(cudaMemcpyAsync(index.bucket_candidate_offsets + n_total_buckets, &n_items_int,
                               sizeof(int), cudaMemcpyHostToDevice, stream));

    // Extract candidate indices (original sample IDs)
    CUDA_CHECK(cudaMalloc(&index.all_candidate_indices, n_items * sizeof(int)));

    extract_sample_ids_kernel<<<grid_size, block_size, 0, stream>>>(
        d_item_indices_sorted, n_samples, n_items, index.all_candidate_indices);

    // Compute table bucket offsets via binary search
    CUDA_CHECK(cudaMalloc(&index.table_bucket_offsets, (n_hash_tables + 1) * sizeof(int)));

    // For each table t, find first bucket with key >= (t << n_projections)
    TableStartKeyFunctor table_key_fn(n_projections);
    auto table_keys_iter =
        thrust::make_transform_iterator(thrust::make_counting_iterator<int>(0), table_key_fn);

    thrust::lower_bound(policy, index.bucket_keys, index.bucket_keys + n_total_buckets,
                        table_keys_iter, table_keys_iter + n_hash_tables,
                        index.table_bucket_offsets);

    // Set sentinel
    CUDA_CHECK(cudaMemcpyAsync(index.table_bucket_offsets + n_hash_tables, &n_total_buckets,
                               sizeof(int), cudaMemcpyHostToDevice, stream));

    // Cleanup
    CUDA_CHECK(cudaFree(d_temp_storage));
    CUDA_CHECK(cudaFree(d_keys_sorted));
    CUDA_CHECK(cudaFree(d_item_indices_sorted));
    CUDA_CHECK(cudaFree(d_bucket_flags));
    CUDA_CHECK(cudaFree(d_bucket_ids));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    return index;
}

} // namespace detail
} // namespace rplsh
} // namespace culsh
