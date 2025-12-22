#pragma once

#include "../candidates.cuh"
#include "../index.cuh"
#include "../utils/utils.cuh"
#include <cstdint>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <cub/device/device_segmented_reduce.cuh>
#include <cub/device/device_select.cuh>
#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

namespace culsh {
namespace rplsh {
namespace detail {

/**
 * @brief Binary search for the given query signature amongst all bucket signatures
 * @param[in] all_bucket_signatures Device pointer to array of all bucket signatures for each table
 * @param[in] table_start Start index of the table's buckets in all_bucket_signatures
 * @param[in] table_end End index of the table's buckets in all_bucket_signatures
 * @param[in] query_sig Device pointer to query signature
 * @param[in] n_projections Number of projections (width of signature)
 * @return Matching bucket index or -1 if no match found
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
 * @param[in] Q_sig Device pointer to query signature matrix
 * @param[in] all_bucket_signatures Device pointer to array of all bucket signatures for each table
 * @param[in] table_bucket_offsets Device pointer to array of offsets for each table's buckets in
 * all_bucket_signatures
 * @param[in] n_queries Number of queries
 * @param[in] n_hash_tables Number of hash tables
 * @param[in] n_projections Number of projections (width of signature)
 * @param[out] matched_bucket_indices Device pointer to array of matching bucket indices for each
 * (query, table) pair
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

    // get start and end of table's buckets in all_bucket_signatures
    int table_start = table_bucket_offsets[table_idx];
    int table_end = table_bucket_offsets[table_idx + 1];

    // look for matching bucket
    int matched_bucket = find_bucket_in_table(all_bucket_signatures, table_start, table_end,
                                              query_sig, n_projections);

    matched_bucket_indices[idx] = matched_bucket;
}

/**
 * @brief Count candidates for each (query, table) pair
 * @param[in] bucket_candidate_offsets Device pointer to array of offsets for each bucket's
 * candidates in all_candidate_indices
 * @param[in] matched_bucket_indices Device pointer to array of matching bucket indices for each
 * (query, table) pair
 * @param[in] n_queries Number of queries
 * @param[in] n_hash_tables Number of hash tables
 * @param[out] candidate_counts Device pointer to array of candidate counts for each (query, table)
 * pair
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
 * @param[in] candidate_counts Device pointer to array of candidate counts for each (query, table)
 * pair
 * @param[in] n_queries Number of queries
 * @param[in] n_hash_tables Number of hash tables
 * @param[out] query_candidate_counts Device pointer to array of candidate counts for each query
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
 * @param[in] candidate_counts Device pointer to array of candidate counts for each (query, table)
 * pair
 * @param[in] n_queries Number of queries
 * @param[in] n_hash_tables Number of hash tables
 * @param[out] table_prefix_offsets Device pointer to array of offsets for each table's candidates
 * for each query
 */
__global__ void compute_table_prefix_offsets_kernel(const int* candidate_counts, int n_queries,
                                                    int n_hash_tables,
                                                    size_t* table_prefix_offsets) {
    int query_id = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (query_id >= n_queries)
        return;

    // this is a segmented prefix sum over the tables for each query,
    // where segments are a fixed n_hash_tables wide.
    size_t running = 0;
    for (int table_id = 0; table_id < n_hash_tables; ++table_id) {
        size_t idx = static_cast<size_t>(query_id) * n_hash_tables + table_id;
        table_prefix_offsets[idx] = running;
        running += static_cast<size_t>(candidate_counts[idx]);
    }
}

/**
 * @brief Collect candidates from each (query, table) pair into an array
 * @param[in] bucket_candidate_offsets Device pointer to array of offsets for each bucket's
 * candidates in all_candidate_indices
 * @param[in] all_candidate_indices Device pointer to array of all candidate indices
 * @param[in] matched_bucket_indices Device pointer to array of matching bucket indices for each
 * (query, table) pair
 * @param[in] query_candidate_offsets Device pointer to array of offsets for each query's candidates
 * @param[in] table_prefix_offsets Device pointer to array of offsets for each table's candidates
 * for each query
 * @param[in] n_queries Number of queries
 * @param[in] n_hash_tables Number of hash tables
 * @param[out] output_candidates Device pointer to array of collected candidates
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
 * @brief Mark unique candidate indices (start of array or different from previous)
 * @param[in] sorted_candidates Device pointer to array of sorted candidate indices
 * @param[in] n_items Number of items
 * @param[out] flags Device pointer to array of flags
 */
__global__ void mark_unique_kernel(const int* sorted_candidates, size_t n_items, int* flags) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n_items)
        return;

    if (idx == 0) {
        flags[idx] = 1;
        return;
    }

    // Mark 1 if current is different from previous, else 0
    flags[idx] = (sorted_candidates[idx] != sorted_candidates[idx - 1]);
}

/**
 * @brief Ensure start of each query segment is marked as 1
 * @param[in] offsets Device pointer to array of offsets
 * @param[in] n_queries Number of queries
 * @param[out] flags Device pointer to array of flags
 */
__global__ void fix_boundaries_kernel(const size_t* offsets, int n_queries, int* flags) {
    int idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n_queries)
        return;

    size_t start = offsets[idx];
    size_t end = offsets[idx + 1];

    // If segment is not empty, force first element to be kept
    if (start < end) {
        flags[start] = 1;
    }
}

/**
 * @brief Query LSH index to find candidates
 * @param[in] stream CUDA stream
 * @param[in] Q_sig Device pointer to query signature matrix (n_queries x n_hash_tables *
 * n_projections)
 * @param[in] n_queries Number of input rows
 * @param[in] n_hash_tables Number of hash tables
 * @param[in] n_projections Number of projections
 * @param[in] index Device index
 * @return Candidates object
 */
Candidates query_index(cudaStream_t stream, const int8_t* Q_sig, int n_queries, int n_hash_tables,
                       int n_projections, const Index* index) {

    size_t n_items = static_cast<size_t>(n_queries) * n_hash_tables;
    dim3 block_size(256);
    dim3 grid_size_items((n_items + block_size.x - 1) / block_size.x);
    dim3 grid_size_queries((n_queries + block_size.x - 1) / block_size.x);

    // For each query, we look at each hash table (aka, column of width n_projections)
    // and find the bucket that matches the query signature (if it exists).
    // For each such matching bucket, all candidates in the index are considered neighbors
    // of the query. Since the candidates across buckets are not disjoint, we deduplicate
    // each query's final collected array of candidates.

    // allocate arrays to store query matches
    int *d_matched_bucket_indices, *d_candidate_counts;
    CUDA_CHECK(cudaMalloc(&d_matched_bucket_indices, n_items * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_candidate_counts, n_items * sizeof(int)));

    // find matching buckets for each (query, table)
    find_matching_buckets_kernel<<<grid_size_items, block_size, 0, stream>>>(
        Q_sig, index->all_bucket_signatures, index->table_bucket_offsets, n_queries, n_hash_tables,
        n_projections, d_matched_bucket_indices);

    // count total number of candidates for each (query, table)
    count_candidates_kernel<<<grid_size_items, block_size, 0, stream>>>(
        index->bucket_candidate_offsets, d_matched_bucket_indices, n_queries, n_hash_tables,
        d_candidate_counts);

    // initialize output candidates object
    Candidates candidates;
    candidates.n_queries = n_queries;
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
    size_t scan_temp_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, scan_temp_bytes,
                                             candidates.query_candidate_counts,
                                             candidates.query_candidate_offsets, n_queries, stream));
    ensure_temp_storage(&d_temp_storage, temp_storage_bytes, scan_temp_bytes);
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                             candidates.query_candidate_counts,
                                             candidates.query_candidate_offsets, n_queries, stream));

    // allocate 'raw' total candidates
    // this allocates space for all candidates collected per query across all tables, including
    // duplicates
    size_t total_candidates_offset;
    size_t last_query_count;
    CUDA_CHECK(cudaMemcpyAsync(&total_candidates_offset,
                               candidates.query_candidate_offsets + (n_queries - 1), sizeof(size_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&last_query_count,
                               candidates.query_candidate_counts + (n_queries - 1), sizeof(size_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));  // sync is needed to set total_raw_candidates on the host

    size_t total_raw_candidates = total_candidates_offset + last_query_count;

    // set terminating value in offsets
    CUDA_CHECK(cudaMemcpyAsync(candidates.query_candidate_offsets + n_queries,
                               &total_raw_candidates, sizeof(size_t), cudaMemcpyHostToDevice,
                               stream));

    if (total_raw_candidates > 0) {
        int* d_raw_candidates;
        CUDA_CHECK(cudaMalloc(&d_raw_candidates, total_raw_candidates * sizeof(int)));

        // collect all candidates (each warp handles one (query, table) pair)
        int warps_per_block = block_size.x >> 5;
        dim3 grid_size_pairs((n_items + warps_per_block - 1) / warps_per_block);
        collect_candidates_kernel<<<grid_size_pairs, block_size, 0, stream>>>(
            index->bucket_candidate_offsets, index->all_candidate_indices, d_matched_bucket_indices,
            candidates.query_candidate_offsets, d_table_prefix_offsets, n_queries, n_hash_tables,
            d_raw_candidates);

        // free temp storage
        CUDA_CHECK(cudaFree(d_matched_bucket_indices));
        CUDA_CHECK(cudaFree(d_candidate_counts));
        CUDA_CHECK(cudaFree(d_table_prefix_offsets));

        // --- deduplicate candidates ---
        // 1. sort the raw candidate indices per-query
        // 2. mark unique candidate indices per segment
        // 3. compute new counts per query
        // 4. compute new offsets
        // 5. get total unique candidates
        // 6. compact unique candidates to final array
        int* d_sorted_candidates;
        CUDA_CHECK(cudaMalloc(&d_sorted_candidates, total_raw_candidates * sizeof(int)));

        // sort the raw candidate indices per-query
        size_t sort_temp_bytes = 0;
        cub::DeviceSegmentedRadixSort::SortKeys(
            nullptr, sort_temp_bytes, d_raw_candidates, d_sorted_candidates, total_raw_candidates,
            n_queries, candidates.query_candidate_offsets, candidates.query_candidate_offsets + 1,
            0, sizeof(int) * 8, stream);

        ensure_temp_storage(&d_temp_storage, temp_storage_bytes, sort_temp_bytes);
        cub::DeviceSegmentedRadixSort::SortKeys(
            d_temp_storage, temp_storage_bytes, d_raw_candidates, d_sorted_candidates,
            total_raw_candidates, n_queries, candidates.query_candidate_offsets,
            candidates.query_candidate_offsets + 1, 0, sizeof(int) * 8, stream);

        // mark unique candidate indices per segment
        // note that this checks global uniqueness - does not distinguish between segment boundaries
        int* d_flags;
        CUDA_CHECK(cudaMalloc(&d_flags, total_raw_candidates * sizeof(int)));

        dim3 grid_size_total((total_raw_candidates + block_size.x - 1) / block_size.x);
        mark_unique_kernel<<<grid_size_total, block_size, 0, stream>>>(
            d_sorted_candidates, total_raw_candidates, d_flags);
        // fix boundaries - post-process segment boundaries to ensure start of each segment is
        // marked as unique
        fix_boundaries_kernel<<<grid_size_queries, block_size, 0, stream>>>(
            candidates.query_candidate_offsets, n_queries, d_flags);

        // compute new candidate counts after de-duplication
        // segmented sum on the marked flags to get new counts per query
        size_t reduce_temp_bytes = 0;
        cub::DeviceSegmentedReduce::Sum(
            nullptr, reduce_temp_bytes, d_flags, candidates.query_candidate_counts, n_queries,
            candidates.query_candidate_offsets, candidates.query_candidate_offsets + 1, stream);

        ensure_temp_storage(&d_temp_storage, temp_storage_bytes, reduce_temp_bytes);

        cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes, d_flags,
                                        candidates.query_candidate_counts, n_queries,
                                        candidates.query_candidate_offsets,
                                        candidates.query_candidate_offsets + 1, stream);

        // compute new offsets via exclusive sum on the new counts
        size_t scan_temp_bytes = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, scan_temp_bytes, candidates.query_candidate_counts,
                                      candidates.query_candidate_offsets, n_queries, stream);
        ensure_temp_storage(&d_temp_storage, temp_storage_bytes, scan_temp_bytes);
        cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                      candidates.query_candidate_counts,
                                      candidates.query_candidate_offsets, n_queries, stream);

        // store total unique candidates to candidates object
        CUDA_CHECK(cudaMemcpyAsync(&total_candidates_offset,
                                   candidates.query_candidate_offsets + (n_queries - 1),
                                   sizeof(size_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(&last_query_count,
                                   candidates.query_candidate_counts + (n_queries - 1),
                                   sizeof(size_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));  // sync is needed to set n_total_candidates on the host

        size_t total_unique_candidates = total_candidates_offset + last_query_count;
        candidates.n_total_candidates = total_unique_candidates;

        // set terminating value
        CUDA_CHECK(cudaMemcpyAsync(candidates.query_candidate_offsets + n_queries,
                                   &total_unique_candidates, sizeof(size_t), cudaMemcpyHostToDevice,
                                   stream));

        // compact unique candidates to final array
        if (total_unique_candidates > 0) {
            CUDA_CHECK(cudaMalloc(&candidates.query_candidate_indices,
                                  total_unique_candidates * sizeof(int)));

            int* d_num_selected_out;
            CUDA_CHECK(cudaMalloc(&d_num_selected_out, sizeof(int)));

            size_t select_temp_bytes = 0;
            cub::DeviceSelect::Flagged(nullptr, select_temp_bytes, d_sorted_candidates, d_flags,
                                       candidates.query_candidate_indices, d_num_selected_out,
                                       total_raw_candidates, stream);

            ensure_temp_storage(&d_temp_storage, temp_storage_bytes, select_temp_bytes);

            cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_sorted_candidates,
                                       d_flags, candidates.query_candidate_indices,
                                       d_num_selected_out, total_raw_candidates, stream);

            CUDA_CHECK(cudaFree(d_num_selected_out));
        } else {
            candidates.query_candidate_indices = nullptr;
        }

        // cleanup
        CUDA_CHECK(cudaFree(d_raw_candidates));
        CUDA_CHECK(cudaFree(d_sorted_candidates));
        CUDA_CHECK(cudaFree(d_flags));

    } else {
        candidates.query_candidate_indices = nullptr;
        candidates.n_total_candidates = 0;

        CUDA_CHECK(cudaFree(d_matched_bucket_indices));
        CUDA_CHECK(cudaFree(d_candidate_counts));
        CUDA_CHECK(cudaFree(d_table_prefix_offsets));
    }

    // cleanup temp
    CUDA_CHECK(cudaFree(d_temp_storage));

    return candidates;
}

} // namespace detail
} // namespace rplsh
} // namespace culsh
