#pragma once

#include "../candidates.cuh"
#include "../index.cuh"
#include "../utils/utils.cuh"
#include <cstdint>
#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

namespace culsh {
namespace rplsh {
namespace detail {

/**
 * @brief Pack query hash values into keys.
 */
__global__ void pack_query_keys_kernel(const float* Q_hash, int n_queries, int n_hash_tables,
                                       int n_projections, size_t n_items, uint32_t* query_keys) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n_items)
        return;

    // idx encodes (query_id * n_hash_tables + table_id)
    int query_id = static_cast<int>(idx / n_hash_tables);
    int table_id = static_cast<int>(idx % n_hash_tables);

    // pack signature bits from hash values
    // layout: [query][table][projection]
    uint32_t sig = 0;
    for (int p = 0; p < n_projections; ++p) {
        size_t hash_idx = static_cast<size_t>(query_id) * n_hash_tables * n_projections +
                          static_cast<size_t>(table_id) * n_projections + p;
        uint32_t bit = (Q_hash[hash_idx] >= 0.0f) ? 1u : 0u;
        sig = (sig << 1) | bit;
    }

    // combine table_id (high bits) with signature (low bits)
    query_keys[idx] = (static_cast<uint32_t>(table_id) << n_projections) | sig;
}

/**
 * @brief Find matching bucket for each (query, table) pair.
 */
__global__ void find_matching_buckets_kernel(const uint32_t* query_keys,
                                             const uint32_t* bucket_keys,
                                             const int* table_bucket_offsets, int n_queries,
                                             int n_hash_tables, size_t n_items,
                                             int* matched_bucket_indices) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n_items)
        return;

    int table_id = static_cast<int>(idx % n_hash_tables);

    uint32_t key = query_keys[idx];
    int table_start = table_bucket_offsets[table_id];
    int table_end = table_bucket_offsets[table_id + 1];

    if (table_start >= table_end) {
        matched_bucket_indices[idx] = -1;
        return;
    }

    // binary search for key in table's bucket range
    int lo = table_start;
    int hi = table_end - 1;
    int result = -1;

    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        uint32_t mid_key = bucket_keys[mid];

        if (mid_key == key) {
            result = mid;
            break;
        } else if (mid_key < key) {
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }

    matched_bucket_indices[idx] = result;
}

/**
 * @brief Count candidates for each (query, table) pair.
 */
__global__ void count_candidates_kernel(const int* bucket_candidate_offsets,
                                        const int* matched_bucket_indices, size_t n_items,
                                        int* candidate_counts) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n_items)
        return;

    int matched_bucket = matched_bucket_indices[idx];
    if (matched_bucket == -1) {
        candidate_counts[idx] = 0;
    } else {
        int start = bucket_candidate_offsets[matched_bucket];
        int end = bucket_candidate_offsets[matched_bucket + 1];
        candidate_counts[idx] = end - start;
    }
}

/**
 * @brief Aggregate candidate counts per query across all tables.
 */
__global__ void aggregate_query_counts_kernel(const int* candidate_counts, int n_queries,
                                              int n_hash_tables, size_t* query_candidate_counts) {
    int query_id = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (query_id >= n_queries)
        return;

    size_t total = 0;
    for (int t = 0; t < n_hash_tables; ++t) {
        total += candidate_counts[static_cast<size_t>(query_id) * n_hash_tables + t];
    }

    query_candidate_counts[query_id] = total;
}

/**
 * @brief Compute per-(query, table) exclusive offsets for candidate collection.
 */
__global__ void compute_table_prefix_offsets_kernel(const int* candidate_counts, int n_queries,
                                                    int n_hash_tables, size_t n_items,
                                                    size_t* table_prefix_offsets) {
    int query_id = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (query_id >= n_queries)
        return;

    size_t running = 0;
    for (int t = 0; t < n_hash_tables; ++t) {
        size_t idx = static_cast<size_t>(query_id) * n_hash_tables + t;
        table_prefix_offsets[idx] = running;
        running += static_cast<size_t>(candidate_counts[idx]);
    }
}

/**
 * @brief Collect candidates into output array.
 */
__global__ void
collect_candidates_kernel(const int* bucket_candidate_offsets, const int* all_candidate_indices,
                          const int* matched_bucket_indices, const size_t* query_candidate_offsets,
                          const size_t* table_prefix_offsets, int n_queries, int n_hash_tables,
                          size_t n_items, int* output_candidates) {
    // one warp per (query, table) pair
    int lane = threadIdx.x & 31;
    int warps_per_block = blockDim.x >> 5;
    size_t pair_index = static_cast<size_t>(blockIdx.x) * warps_per_block + (threadIdx.x >> 5);

    if (pair_index >= n_items)
        return;

    int query_idx = static_cast<int>(pair_index / n_hash_tables);
    int matched_bucket = matched_bucket_indices[pair_index];

    if (matched_bucket == -1)
        return;

    // get bucket span
    int bucket_start = bucket_candidate_offsets[matched_bucket];
    int bucket_end = bucket_candidate_offsets[matched_bucket + 1];
    int bucket_size = bucket_end - bucket_start;

    if (bucket_size <= 0)
        return;

    // output position
    size_t base_out = query_candidate_offsets[query_idx] + table_prefix_offsets[pair_index];
    for (int i = lane; i < bucket_size; i += 32) {
        output_candidates[base_out + i] = all_candidate_indices[bucket_start + i];
    }
}

/**
 * @brief Query LSH index to find candidates.
 *
 * @param[in] stream CUDA stream
 * @param[in] Q_hash Query hash values (n_queries x n_hash_tables * n_projections)
 * @param[in] n_queries Number of queries
 * @param[in] n_hash_tables Number of hash tables
 * @param[in] n_projections Number of projections per table
 * @param[in] index Device index
 * @return Candidates structure with results
 */
Candidates query_index(cudaStream_t stream, const float* Q_hash, int n_queries, int n_hash_tables,
                       int n_projections, const Index* index) {

    auto policy = thrust::cuda::par.on(stream);
    size_t n_items = static_cast<size_t>(n_queries) * n_hash_tables;

    dim3 block_size(256);
    dim3 grid_size_items((n_items + block_size.x - 1) / block_size.x);
    dim3 grid_size_queries((n_queries + block_size.x - 1) / block_size.x);

    // pack query hash values into keys
    uint32_t* d_query_keys;
    CUDA_CHECK(cudaMalloc(&d_query_keys, n_items * sizeof(uint32_t)));

    pack_query_keys_kernel<<<grid_size_items, block_size, 0, stream>>>(
        Q_hash, n_queries, n_hash_tables, n_projections, n_items, d_query_keys);

    // find matching bucket for each (query, table)
    int* d_matched_buckets;
    CUDA_CHECK(cudaMalloc(&d_matched_buckets, n_items * sizeof(int)));

    find_matching_buckets_kernel<<<grid_size_items, block_size, 0, stream>>>(
        d_query_keys, index->bucket_keys, index->table_bucket_offsets, n_queries, n_hash_tables,
        n_items, d_matched_buckets);

    // count candidates per (query, table)
    int* d_candidate_counts;
    CUDA_CHECK(cudaMalloc(&d_candidate_counts, n_items * sizeof(int)));

    count_candidates_kernel<<<grid_size_items, block_size, 0, stream>>>(
        index->bucket_candidate_offsets, d_matched_buckets, n_items, d_candidate_counts);

    // aggregate counts per query
    Candidates candidates;
    CUDA_CHECK(cudaMalloc(&candidates.query_candidate_counts, n_queries * sizeof(size_t)));

    aggregate_query_counts_kernel<<<grid_size_queries, block_size, 0, stream>>>(
        d_candidate_counts, n_queries, n_hash_tables, candidates.query_candidate_counts);

    // compute output offsets via prefix sum
    CUDA_CHECK(cudaMalloc(&candidates.query_candidate_offsets, (n_queries + 1) * sizeof(size_t)));

    thrust::exclusive_scan(policy, candidates.query_candidate_counts,
                           candidates.query_candidate_counts + n_queries,
                           candidates.query_candidate_offsets, size_t(0));

    // get total candidates
    size_t total_candidates_offset, last_query_count;
    CUDA_CHECK(cudaMemcpyAsync(&total_candidates_offset,
                               candidates.query_candidate_offsets + (n_queries - 1), sizeof(size_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&last_query_count,
                               candidates.query_candidate_counts + (n_queries - 1), sizeof(size_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    size_t total_candidates = total_candidates_offset + last_query_count;

    printf("Total candidates: %zu (%.2f GB)\n", total_candidates,
           (double)total_candidates * sizeof(int) / (1024.0 * 1024.0 * 1024.0));

    // set sentinel value
    CUDA_CHECK(cudaMemcpyAsync(candidates.query_candidate_offsets + n_queries, &total_candidates,
                               sizeof(size_t), cudaMemcpyHostToDevice, stream));

    // collect candidates
    if (total_candidates > 0) {
        CUDA_CHECK(cudaMalloc(&candidates.query_candidate_indices, total_candidates * sizeof(int)));

        // compute per-table prefix offsets
        size_t* d_table_prefix_offsets;
        CUDA_CHECK(cudaMalloc(&d_table_prefix_offsets, n_items * sizeof(size_t)));

        compute_table_prefix_offsets_kernel<<<grid_size_queries, block_size, 0, stream>>>(
            d_candidate_counts, n_queries, n_hash_tables, n_items, d_table_prefix_offsets);

        // collection candidates
        int warps_per_block = block_size.x >> 5;
        int grid_size_pairs = static_cast<int>((n_items + warps_per_block - 1) / warps_per_block);

        collect_candidates_kernel<<<grid_size_pairs, block_size, 0, stream>>>(
            index->bucket_candidate_offsets, index->all_candidate_indices, d_matched_buckets,
            candidates.query_candidate_offsets, d_table_prefix_offsets, n_queries, n_hash_tables,
            n_items, candidates.query_candidate_indices);

        CUDA_CHECK(cudaFree(d_table_prefix_offsets));
    } else {
        candidates.query_candidate_indices = nullptr;
    }

    // cleanup
    CUDA_CHECK(cudaFree(d_query_keys));
    CUDA_CHECK(cudaFree(d_matched_buckets));
    CUDA_CHECK(cudaFree(d_candidate_counts));

    return candidates;
}

} // namespace detail
} // namespace rplsh
} // namespace culsh
