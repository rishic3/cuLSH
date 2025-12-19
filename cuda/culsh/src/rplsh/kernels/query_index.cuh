#pragma once

#include "../candidates.cuh"
#include "../index.cuh"
#include "../utils/utils.cuh"
#include <cstdint>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

namespace culsh {
namespace rplsh {
namespace detail {

/**
 * @brief Binary search for query signature amongst buckets of a hash table
 */
__device__ int find_bucket_in_table(const int8_t* all_bucket_signatures, int table_start,
                                    int table_end, const int8_t* query_sig, int n_projections) {

    if (table_start >= table_end)
        return -1;

    // binary search within table buckets
    int lo = table_start;
    int hi = table_end - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        const int8_t* mid_sig = all_bucket_signatures + static_cast<size_t>(mid) * n_projections;

        // check if signature matches
        int cmp = 0;
        for (int i = 0; i < n_projections && cmp == 0; ++i) {
            cmp = (query_sig[i] > mid_sig[i]) - (query_sig[i] < mid_sig[i]);
        }

        if (cmp == 0) {
            return mid;
        } else if (cmp < 0) {
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
    }
    return -1;
}

/**
 * @brief Find matching bucket for each (query, table) pair
 */
__global__ void find_matching_buckets_kernel(const int8_t* Q_sig,
                                             const int8_t* all_bucket_signatures,
                                             const int* table_bucket_offsets, int n_queries,
                                             int n_hash_tables, int n_projections,
                                             int* matched_bucket_indices) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t n_items = static_cast<size_t>(n_queries) * n_hash_tables;
    if (idx >= n_items)
        return;

    int query_idx = idx / n_hash_tables;
    int table_idx = idx % n_hash_tables;

    // get pointer to query signature in Q_sig
    const int8_t* query_sig = Q_sig + static_cast<size_t>(table_idx) * (n_queries * n_projections) +
                              static_cast<size_t>(query_idx) * n_projections;

    // start and end of table's buckets in all_bucket_signatures
    int table_start = table_bucket_offsets[table_idx];
    int table_end = table_bucket_offsets[table_idx + 1];

    // look for matching bucket
    int matched_bucket = find_bucket_in_table(all_bucket_signatures, table_start, table_end,
                                              query_sig, n_projections);

    matched_bucket_indices[idx] = matched_bucket;
}

/**
 * @brief Count candidates for each (query, table) pair
 */
__global__ void count_candidates_kernel(const int* bucket_candidate_offsets,
                                        const int* matched_bucket_indices, int n_queries,
                                        int n_hash_tables, int* candidate_counts) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t n_items = static_cast<size_t>(n_queries) * n_hash_tables;
    if (idx >= n_items)
        return;

    int matched_bucket = matched_bucket_indices[idx];
    if (matched_bucket == -1) {
        candidate_counts[idx] = 0;
    } else {
        // get start and end of bucket's candidates in all_candidate_indices
        int start = bucket_candidate_offsets[matched_bucket];
        int end = bucket_candidate_offsets[matched_bucket + 1];
        candidate_counts[idx] = end - start; // num candidates = size of that range
    }
}

/**
 * @brief Aggregate candidates per query across all tables
 */
__global__ void aggregate_query_results_kernel(const int* candidate_counts, int n_queries,
                                               int n_hash_tables, size_t* query_candidate_counts) {
    int query_id = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (query_id >= n_queries)
        return;

    // sum candidates across all tables for this query
    size_t total_candidates = 0;
    for (int table_id = 0; table_id < n_hash_tables; ++table_id) {
        size_t idx = static_cast<size_t>(query_id) * n_hash_tables + table_id;
        total_candidates += candidate_counts[idx];
    }

    query_candidate_counts[query_id] = total_candidates;
}

/**
 * @brief Precompute per-(query, table) exclusive offsets for candidate copies
 */
__global__ void compute_table_prefix_offsets_kernel(const int* candidate_counts, int n_queries,
                                                    int n_hash_tables,
                                                    size_t* table_prefix_offsets) {
    int query_id = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (query_id >= n_queries)
        return;

    // for each query, compute exclusive prefix over tables t
    size_t running = 0;
    for (int table_id = 0; table_id < n_hash_tables; ++table_id) {
        size_t idx = static_cast<size_t>(query_id) * n_hash_tables + table_id;
        table_prefix_offsets[idx] = running;
        running += static_cast<size_t>(candidate_counts[idx]);
    }
}

/**
 * @brief Collect candidates into final output array
 */
__global__ void collect_candidates_kernel(const int* bucket_candidate_offsets,
                                          const int* all_candidate_indices,
                                          const int* matched_bucket_indices,
                                          const size_t* query_candidate_offsets,
                                          const size_t* table_prefix_offsets, int n_queries,
                                          int n_hash_tables, int* output_candidates) {
    // one warp per (query, table) pair
    int lane = threadIdx.x & 31;
    int warps_per_block = blockDim.x >> 5;
    size_t pair_index = static_cast<size_t>(blockIdx.x) * warps_per_block + (threadIdx.x >> 5);
    size_t n_items = static_cast<size_t>(n_queries) * n_hash_tables;
    if (pair_index >= n_items)
        return;

    // load matched bucket id for query
    int query_idx = static_cast<int>(pair_index / n_hash_tables);
    int table_idx = static_cast<int>(pair_index % n_hash_tables);
    int matched_bucket = matched_bucket_indices[pair_index];
    if (matched_bucket == -1)
        return;

    // get bucket span from bucket offsets in candidates array
    int bucket_start = bucket_candidate_offsets[matched_bucket];
    int bucket_end = bucket_candidate_offsets[matched_bucket + 1];
    int bucket_size = bucket_end - bucket_start;
    if (bucket_size <= 0)
        return;

    // base idx to write candidates for this bucket
    size_t base_out =
        query_candidate_offsets[query_idx] +
        table_prefix_offsets[static_cast<size_t>(query_idx) * n_hash_tables + table_idx];

    // stride by warp size for coalesced access
    for (int i = lane; i < bucket_size; i += 32) {
        output_candidates[base_out + i] = all_candidate_indices[bucket_start + i];
    }
}

/**
 * @brief Query LSH index to find candidates
 * @param[in] stream CUDA stream
 * @param[in] Q_sig signature matrix (n_queries x n_hash_tables * n_projections)
 * @param[in] n_queries Number of input rows
 * @param[in] n_hash_tables Number of hash tables
 * @param[in] n_projections Number of projections
 * @param[in] index Device index
 * @param[out] candidates Candidate indices for each query.
 */
Candidates query_index(cudaStream_t stream, const int8_t* Q_sig, int n_queries, int n_hash_tables,
                       int n_projections, const Index* index) {

    size_t n_items = static_cast<size_t>(n_queries) * n_hash_tables;
    dim3 block_size(256);
    dim3 grid_size_items((n_items + block_size.x - 1) / block_size.x);
    dim3 grid_size_queries((n_queries + block_size.x - 1) / block_size.x);

    // allocate arrays to store query matches
    int *d_matched_bucket_indices, *d_candidate_counts;
    CUDA_CHECK(cudaMalloc(&d_matched_bucket_indices, n_items * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_candidate_counts, n_items * sizeof(int)));

    // find matching buckets for each (query, table)
    find_matching_buckets_kernel<<<grid_size_items, block_size, 0, stream>>>(
        Q_sig, index->all_bucket_signatures, index->table_bucket_offsets, n_queries, n_hash_tables,
        n_projections, d_matched_bucket_indices);

    // count candidate for each (query, table)
    count_candidates_kernel<<<grid_size_items, block_size, 0, stream>>>(
        index->bucket_candidate_offsets, d_matched_bucket_indices, n_queries, n_hash_tables,
        d_candidate_counts);

    // initialize output candidates
    Candidates candidates;
    CUDA_CHECK(cudaMalloc(&candidates.query_candidate_counts, n_queries * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&candidates.query_candidate_offsets, (n_queries + 1) * sizeof(size_t)));

    // for each query, sum candidates across all tables
    aggregate_query_results_kernel<<<grid_size_queries, block_size, 0, stream>>>(
        d_candidate_counts, n_queries, n_hash_tables, candidates.query_candidate_counts);

    // precompute per-(query, table) write offsets
    size_t* d_table_prefix_offsets;
    CUDA_CHECK(cudaMalloc(&d_table_prefix_offsets,
                          static_cast<size_t>(n_queries) * n_hash_tables * sizeof(size_t)));
    compute_table_prefix_offsets_kernel<<<grid_size_queries, block_size, 0, stream>>>(
        d_candidate_counts, n_queries, n_hash_tables, d_table_prefix_offsets);

    // compute prefix sum for query offsets
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                  candidates.query_candidate_counts,
                                  candidates.query_candidate_offsets, n_queries, stream);
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                  candidates.query_candidate_counts,
                                  candidates.query_candidate_offsets, n_queries, stream);

    // allocate total candidates
    size_t total_candidates_offset;
    size_t last_query_count;
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

    // set terminating value in offsets
    CUDA_CHECK(cudaMemcpyAsync(candidates.query_candidate_offsets + n_queries, &total_candidates,
                               sizeof(size_t), cudaMemcpyHostToDevice, stream));

    if (total_candidates > 0) {
        CUDA_CHECK(cudaMalloc(&candidates.query_candidate_indices, total_candidates * sizeof(int)));

        // collect candidates (each warp handles one (query, table) pair)
        int warps_per_block = block_size.x >> 5;
        dim3 grid_size_pairs((n_items + warps_per_block - 1) / warps_per_block);
        collect_candidates_kernel<<<grid_size_pairs, block_size, 0, stream>>>(
            index->bucket_candidate_offsets, index->all_candidate_indices, d_matched_bucket_indices,
            candidates.query_candidate_offsets, d_table_prefix_offsets, n_queries, n_hash_tables,
            candidates.query_candidate_indices);
    } else {
        candidates.query_candidate_indices = nullptr;
    }

    // cleanup
    CUDA_CHECK(cudaFree(d_temp_storage));
    CUDA_CHECK(cudaFree(d_table_prefix_offsets));
    CUDA_CHECK(cudaFree(d_matched_bucket_indices));
    CUDA_CHECK(cudaFree(d_candidate_counts));

    return candidates;
}

} // namespace detail
} // namespace rplsh
} // namespace culsh
