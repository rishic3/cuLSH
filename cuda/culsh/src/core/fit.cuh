#pragma once

#include "constants.cuh"
#include "index.cuh"
#include "utils.cuh"
#include <cassert>
#include <cstdint>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

namespace culsh {
namespace core {
namespace detail {

/**
 * @brief Extract n-th byte of signature for radix sort
 * @param[in] X_sig Device pointer to signature matrix (n_hash_tables x n_samples x sig_nbytes)
 * @param[in] item_indices Device pointer to array of item indices
 * @param[in] n_samples Number of input rows
 * @param[in] n_hash_tables Number of hash tables
 * @param[in] sig_nbytes Signature width in bytes
 * @param[in] byte_idx Index of byte to extract
 * @param[out] d_keys Device pointer to array of keys
 */
static __global__ void extract_byte_key_kernel(const uint8_t* X_sig, const uint32_t* item_indices,
                                               int n_samples, int n_hash_tables, int sig_nbytes,
                                               int byte_idx, uint8_t* d_keys) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t n_items = static_cast<size_t>(n_samples) * n_hash_tables;
    if (idx >= n_items)
        return;

    // Each thread idx handles one item
    // Get nth byte of signature for item_indices[idx]
    uint32_t original_item_idx = item_indices[idx];
    uint32_t table_id = original_item_idx / n_samples;
    uint32_t row_id = original_item_idx % n_samples;

    // Get pointer to original signature in X_sig
    const uint8_t* sig_ptr =
        X_sig + static_cast<size_t>(table_id) * (static_cast<size_t>(n_samples) * sig_nbytes) +
        static_cast<size_t>(row_id) * sig_nbytes;

    d_keys[idx] = static_cast<uint8_t>(sig_ptr[byte_idx]);
}

/**
 * @brief Extract table id for final radix sort
 * @param[in] item_indices Device pointer to array of item indices
 * @param[in] n_samples Number of input rows
 * @param[in] n_items Number of items (n_samples * n_hash_tables)
 * @param[out] d_keys Device pointer to array of keys
 */
static __global__ void extract_table_id_key_kernel(const uint32_t* item_indices, int n_samples,
                                                   size_t n_items, uint8_t* d_keys) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n_items)
        return;

    uint32_t original_item_idx = item_indices[idx];
    d_keys[idx] = static_cast<uint8_t>(original_item_idx / n_samples);
}

/**
 * @brief Check if two signatures are equal
 * @param[in] sig1 Device pointer to first signature
 * @param[in] sig2 Device pointer to second signature
 * @param[in] sig_nbytes Signature width in bytes
 * @return True if signatures are equal, false otherwise
 */
static __device__ bool are_signatures_equal(const uint8_t* sig1, const uint8_t* sig2,
                                            int sig_nbytes) {
    for (int i = 0; i < sig_nbytes; ++i) {
        if (sig1[i] != sig2[i])
            return false;
    }
    return true;
}

/**
 * @brief Mark first item of each unique bucket and each new table after sort
 * @param[in] X_sig Device pointer to signature matrix (n_hash_tables x n_samples x sig_nbytes)
 * @param[in] sorted_item_indices Device pointer to array of sorted item indices
 * @param[in] n_samples Number of input rows
 * @param[in] n_hash_tables Number of hash tables
 * @param[in] sig_nbytes Signature width in bytes
 * @param[in] n_items Number of items (n_samples * n_hash_tables)
 * @param[out] d_bucket_flags Device pointer to array of bucket flags
 * @param[out] d_table_flags Device pointer to array of table flags
 */
static __global__ void mark_boundaries_kernel(const uint8_t* X_sig,
                                              const uint32_t* sorted_item_indices, int n_samples,
                                              int n_hash_tables, int sig_nbytes, size_t n_items,
                                              int* d_bucket_flags, int* d_table_flags) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n_items)
        return;

    // Get current item idx and table id
    uint32_t orig_idx_curr = sorted_item_indices[idx];
    uint32_t table_id_curr = orig_idx_curr / n_samples;

    if (idx == 0) {
        // First item is a new bucket/table
        d_bucket_flags[idx] = 1;
        d_table_flags[idx] = 1;
        return;
    }

    // Get previous item idx and table id
    uint32_t orig_idx_prev = sorted_item_indices[idx - 1];
    uint32_t table_id_prev = orig_idx_prev / n_samples;

    // If table id changed, mark new table
    int table_changed = (table_id_curr != table_id_prev);

    int signature_changed = 0;
    if (!table_changed) {
        // Check if signature changed from prev to curr
        uint32_t row_id_curr = orig_idx_curr % n_samples;
        uint32_t row_id_prev = orig_idx_prev % n_samples;
        const uint8_t* sig_ptr_curr =
            X_sig +
            static_cast<size_t>(table_id_curr) * (static_cast<size_t>(n_samples) * sig_nbytes) +
            static_cast<size_t>(row_id_curr) * sig_nbytes;
        const uint8_t* sig_ptr_prev =
            X_sig +
            static_cast<size_t>(table_id_prev) * (static_cast<size_t>(n_samples) * sig_nbytes) +
            static_cast<size_t>(row_id_prev) * sig_nbytes;
        // If signature changed, mark new bucket
        signature_changed = !are_signatures_equal(sig_ptr_curr, sig_ptr_prev, sig_nbytes);
    }

    d_table_flags[idx] = table_changed;
    d_bucket_flags[idx] = table_changed | signature_changed;
}

/**
 * @brief Scatter sorted data into final flat index
 * @param[in] X_sig Device pointer to signature matrix (n_hash_tables x n_samples x sig_nbytes)
 * @param[in] sorted_item_indices Device pointer to array of sorted item indices
 * @param[in] d_bucket_flags Device pointer to array of bucket flags
 * @param[in] d_bucket_scan Device pointer to array of exclusive scan on bucket flags
 * @param[in] n_samples Number of input rows
 * @param[in] n_hash_tables Number of hash tables
 * @param[in] sig_nbytes Signature width in bytes
 * @param[in] n_items Number of items (n_samples * n_hash_tables)
 * @param[out] d_bucket_signatures Device pointer to array of bucket signatures
 * @param[out] d_bucket_candidate_offsets Device pointer to array of bucket candidate offsets
 * @param[out] d_all_candidates Device pointer to array of all candidate indices
 * @param[out] d_item_to_bucket Optional device pointer containing (row,table) -> bucket_id mapping
 */
static __global__ void build_final_index_kernel(const uint8_t* X_sig,
                                                const uint32_t* sorted_item_indices,
                                                const int* d_bucket_flags, const int* d_bucket_scan,
                                                int n_samples, int n_hash_tables, int sig_nbytes,
                                                size_t n_items, uint8_t* d_bucket_signatures,
                                                int* d_bucket_candidate_offsets,
                                                int* d_all_candidates, int* d_item_to_bucket) {

    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n_items)
        return;

    uint32_t orig_idx = sorted_item_indices[idx];
    uint32_t orig_row_id = orig_idx % n_samples;
    uint32_t table_id = orig_idx / n_samples;

    // Each thread (each item) stores its original row idx to all_candidate_indices in sorted order
    d_all_candidates[idx] = orig_row_id;

    // Optionally store mapping from (row,table) -> bucket_id for this item
    // This directly materializes the equivalent of matched_bucket_indices for fit_query
    if (d_item_to_bucket != nullptr) {
        // Since d_bucket_scan is exclusive, bucket_id = d_bucket_flags[idx] - 1 for items that
        // continue a bucket (for items that start a bucket, bucket_id = d_bucket_scan[idx])
        int bucket_id = d_bucket_scan[idx] + d_bucket_flags[idx] - 1;
        d_item_to_bucket[orig_row_id * n_hash_tables + table_id] = bucket_id;
    }

    if (d_bucket_flags[idx] == 1) {
        // If this thread (item) marks start of a new bucket:
        // 1. write idx to bucket_candidate_offsets (store starting pos of this bucket's candidates)
        // 2. write signature for this bucket to d_bucket_signatures

        int bucket_idx = d_bucket_scan[idx];
        d_bucket_candidate_offsets[bucket_idx] = idx;

        // Find src signature address:
        // (X_sig addr) + table_id * (size of table) + row_id * (size of signature)
        const uint8_t* sig_src_ptr =
            X_sig + static_cast<size_t>(table_id) * (static_cast<size_t>(n_samples) * sig_nbytes) +
            static_cast<size_t>(orig_row_id) * sig_nbytes;
        // Find dst signature address:
        // (d_bucket_signatures addr) + bucket_idx * (size of signature)
        uint8_t* sig_dst_ptr = d_bucket_signatures + static_cast<size_t>(bucket_idx) * sig_nbytes;

        for (int i = 0; i < sig_nbytes; ++i) {
            sig_dst_ptr[i] = sig_src_ptr[i];
        }
    }
}

/**
 * @brief Fit flat LSH index structure on GPU
 * @param[in] stream CUDA stream
 * @param[in] X_sig Device pointer to signature matrix (n_hash_tables x n_samples x sig_nbytes).
 * Treated as opaque bytes and can be any fixed-width type (only byte-level equality matters).
 * @param[in] n_samples Number of input rows
 * @param[in] n_hash_tables Number of hash tables
 * @param[in] n_hashes Number of hashes per table
 * @param[in] sig_nbytes Signature width in bytes
 * @param[out] d_item_to_bucket Optional device pointer containing (table,row) -> bucket_id mapping
 * @return Index object
 */
inline Index fit_index(cudaStream_t stream, const void* X_sig, int n_samples, int n_hash_tables,
                       int n_hashes, int sig_nbytes, int* d_item_to_bucket = nullptr) {
    assert(sig_nbytes > 0 && "sig_nbytes must be > 0");
    const uint8_t* X_sig_bytes = static_cast<const uint8_t*>(X_sig);

    size_t n_items = static_cast<size_t>(n_samples) * n_hash_tables;
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size((n_items + block_size.x - 1) / block_size.x);

    // Create a searchable index structure from the given signature matrix.
    // The index structure is a flat array of all candidates, sorted first by table (aka
    // column of width n_hashes), then lexicographically by signature within each table.
    // (In this sense the flat array is 'column-major' in the sense that tables are stored
    // contiguously.) Note that signatures are unique within each table, but not across tables.

    // Allocate temp storage
    uint32_t* d_item_indices = nullptr;
    uint32_t* d_temp_indices = nullptr;
    uint8_t* d_keys = nullptr;
    uint8_t* d_temp_keys = nullptr;
    int* d_bucket_flags = nullptr;
    int* d_table_flags = nullptr;
    int* d_bucket_scan = nullptr;
    CUDA_CHECK_ALLOC(cudaMalloc(&d_item_indices, n_items * sizeof(uint32_t)));
    CUDA_CHECK_ALLOC(cudaMalloc(&d_temp_indices, n_items * sizeof(uint32_t)));
    CUDA_CHECK_ALLOC(cudaMalloc(&d_keys, n_items * sizeof(uint8_t)));
    CUDA_CHECK_ALLOC(cudaMalloc(&d_temp_keys, n_items * sizeof(uint8_t)));
    CUDA_CHECK_ALLOC(cudaMalloc(&d_bucket_flags, n_items * sizeof(int)));
    CUDA_CHECK_ALLOC(cudaMalloc(&d_table_flags, n_items * sizeof(int)));
    CUDA_CHECK_ALLOC(cudaMalloc(&d_bucket_scan, n_items * sizeof(int)));

    // Initialize sequence d_item_indices [0, 1, 2, ... n_items-1]
    thrust::sequence(thrust::cuda::par.on(stream), d_item_indices, d_item_indices + n_items, 0);

    // Query memory requirements for radix sort
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    size_t sort_temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, sort_temp_bytes, d_keys, d_temp_keys, d_item_indices,
                                    d_temp_indices, n_items, 0, 8, stream);
    ensure_temp_storage(&d_temp_storage, temp_storage_bytes, sort_temp_bytes);

    // Sort items lexicographically by signature, from least -> most significant signature byte
    for (int byte_idx = sig_nbytes - 1; byte_idx >= 0; --byte_idx) {
        // Extract nth byte for each item from corresponding signature in X_sig
        extract_byte_key_kernel<<<grid_size, block_size, 0, stream>>>(
            X_sig_bytes, d_item_indices, n_samples, n_hash_tables, sig_nbytes, byte_idx, d_keys);
        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_temp_keys,
                                        d_item_indices, d_temp_indices, n_items, 0, 8, stream);
        std::swap(d_item_indices, d_temp_indices);
    }

    // Final sort by table id s.t. item groups are sorted by table:
    // d_item_indices = [table_0_item_0, table_0_item_1, ..., table_0_item_n-1,
    //                  table_1_item_0, table_1_item_1, ..., table_1_item_n-1,
    //                  ...]
    extract_table_id_key_kernel<<<grid_size, block_size, 0, stream>>>(d_item_indices, n_samples,
                                                                      n_items, d_keys);
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_temp_keys,
                                    d_item_indices, d_temp_indices, n_items, 0, 8, stream);
    std::swap(d_item_indices, d_temp_indices);

    // Mark boundaries between buckets (groups of unique items) and tables
    mark_boundaries_kernel<<<grid_size, block_size, 0, stream>>>(
        X_sig_bytes, d_item_indices, n_samples, n_hash_tables, sig_nbytes, n_items, d_bucket_flags,
        d_table_flags);

    // Free sort buffers
    CUDA_CHECK(cudaFree(d_keys));
    CUDA_CHECK(cudaFree(d_temp_keys));
    CUDA_CHECK(cudaFree(d_temp_indices));

    // Initialize index
    Index index;
    index.n_hash_tables = n_hash_tables;
    index.n_hashes = n_hashes;
    index.sig_nbytes = sig_nbytes;
    index.n_total_candidates = static_cast<int>(n_items);
    CUDA_CHECK_ALLOC(cudaMalloc(&index.table_bucket_offsets, (n_hash_tables + 1) * sizeof(int)));

    // Run exclusive sum to get bucket offsets from binary flags, e.g.
    // d_bucket_flags = [1, 0, 0, 1, 0, 0, 1, 0, ...]
    // d_bucket_scan = [0, 1, 1, 2, 2, 2, 3, 3, ...]
    size_t scan_temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, scan_temp_bytes, d_bucket_flags, d_bucket_scan, n_items,
                                  stream);
    ensure_temp_storage(&d_temp_storage, temp_storage_bytes, scan_temp_bytes);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_bucket_flags, d_bucket_scan,
                                  n_items, stream);

    int* d_num_selected_out = nullptr;
    CUDA_CHECK_ALLOC(cudaMalloc(&d_num_selected_out, sizeof(int)));

    // Query memory requirements for select
    size_t select_temp_storage_bytes = 0;
    cub::DeviceSelect::Flagged(nullptr, select_temp_storage_bytes, d_bucket_scan, d_table_flags,
                               index.table_bucket_offsets, d_num_selected_out, n_items, stream);
    ensure_temp_storage(&d_temp_storage, temp_storage_bytes, select_temp_storage_bytes);

    // Select from d_bucket_scan using d_table_flags to get table offsets. e.g.
    // d_bucket_scan = [0, 1, 1, 2, 2, 2, 3, 3, ...]
    // d_table_flags = [1, 0, 0, 1, 0, 0, 1, 0, ...]
    // table_bucket_offsets = [0, 2, 3, ...]
    cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_bucket_scan, d_table_flags,
                               index.table_bucket_offsets, d_num_selected_out, n_items, stream);

    // Get total number of unique buckets:
    // last_bucket_idx_val (buckets seen before last item) + last_bucket_flag_val (whether last
    // bucket is new bucket)
    int last_bucket_idx_val, last_bucket_flag_val;
    CUDA_CHECK(cudaMemcpyAsync(&last_bucket_idx_val, d_bucket_scan + (n_items - 1), sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&last_bucket_flag_val, d_bucket_flags + (n_items - 1), sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream)); // sync is needed to set n_total_buckets on the host
    index.n_total_buckets = last_bucket_idx_val + last_bucket_flag_val;

    // Allocate all_candidate_indices - original row idx for all items in sorted order
    CUDA_CHECK_ALLOC(cudaMalloc(&index.all_candidate_indices, n_items * sizeof(int)));
    // Allocate bucket_candidate_offsets - start idx of each bucket's candidate indices in
    // all_candidate_indices
    CUDA_CHECK(
        cudaMalloc(&index.bucket_candidate_offsets, (index.n_total_buckets + 1) * sizeof(int)));
    // Allocate all_bucket_signatures - signature for all buckets in sorted order
    CUDA_CHECK_ALLOC(cudaMalloc(&index.all_bucket_signatures, static_cast<size_t>(index.n_total_buckets) *
                                                            sig_nbytes * sizeof(uint8_t)));

    // Set terminating values in offset arrays
    int n_items_int = static_cast<int>(n_items);
    CUDA_CHECK(cudaMemcpyAsync(index.table_bucket_offsets + n_hash_tables, &index.n_total_buckets,
                               sizeof(int), cudaMemcpyHostToDevice,
                               stream)); // for table_bucket_offsets, set total number of buckets
    CUDA_CHECK(cudaMemcpyAsync(index.bucket_candidate_offsets + index.n_total_buckets, &n_items_int,
                               sizeof(int), cudaMemcpyHostToDevice,
                               stream)); // for bucket_candidate_offsets, set total number of items

    // Scatter sorted data into final flat index
    build_final_index_kernel<<<grid_size, block_size, 0, stream>>>(
        X_sig_bytes, d_item_indices, d_bucket_flags, d_bucket_scan, n_samples, n_hash_tables,
        sig_nbytes, n_items, index.all_bucket_signatures, index.bucket_candidate_offsets,
        index.all_candidate_indices, d_item_to_bucket);

    // Cleanup
    CUDA_CHECK(cudaFree(d_temp_storage));
    CUDA_CHECK(cudaFree(d_item_indices));
    CUDA_CHECK(cudaFree(d_bucket_flags));
    CUDA_CHECK(cudaFree(d_table_flags));
    CUDA_CHECK(cudaFree(d_bucket_scan));
    CUDA_CHECK(cudaFree(d_num_selected_out));

    return index;
}

} // namespace detail
} // namespace core
} // namespace culsh
