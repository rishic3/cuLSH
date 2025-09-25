#pragma once

#include "../index.cuh"
#include "../utils/utils.cuh"
#include <cstdint>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <cuda_runtime.h>

namespace culsh {
namespace rplsh {
namespace detail {

/**
 * @brief Extract n-th byte of signature for radix sort
 */
__global__ void extract_byte_key_kernel(const int8_t* X_sig, const uint32_t* item_indices,
                                        int n_samples, int n_hash_tables, int n_projections,
                                        int byte_idx, uint8_t* d_keys) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t n_items = static_cast<size_t>(n_samples) * n_hash_tables;
    if (idx >= n_items)
        return;

    // each thread idx handles one item
    // get nth byte of signature for item_indices[idx]
    uint32_t original_item_idx = item_indices[idx];
    uint32_t table_id = original_item_idx / n_samples;
    uint32_t row_id = original_item_idx % n_samples;

    // get pointer to original signature in X_sig
    const int8_t* sig_ptr =
        X_sig + table_id * (static_cast<size_t>(n_samples) * n_projections) + row_id * n_projections;

    d_keys[idx] = static_cast<uint8_t>(sig_ptr[byte_idx]);
}

/**
 * @brief Extract table id for final radix sort
 */
__global__ void extract_table_id_key_kernel(const uint32_t* item_indices, int n_samples,
                                            size_t n_items, uint8_t* d_keys) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n_items)
        return;

    uint32_t original_item_idx = item_indices[idx];
    d_keys[idx] = static_cast<uint8_t>(original_item_idx / n_samples);
}

/**
 * @brief Check if two signatures are equal
 */
__device__ bool are_signatures_equal(const int8_t* sig1, const int8_t* sig2, int n) {
    for (int i = 0; i < n; ++i) {
        if (sig1[i] != sig2[i])
            return false;
    }
    return true;
}

/**
 * @brief Mark first item of each unique bucket and each new table after sort
 */
__global__ void mark_boundaries_kernel(const int8_t* X_sig, const uint32_t* sorted_item_indices,
                                       int n_samples, int n_hash_tables, int n_projections,
                                       size_t n_items, int* d_bucket_flags, int* d_table_flags) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n_items)
        return;

    // get current item idx and table id
    uint32_t orig_idx_curr = sorted_item_indices[idx];
    uint32_t table_id_curr = orig_idx_curr / n_samples;

    if (idx == 0) {
        // first item is a new bucket/table
        d_bucket_flags[idx] = 1;
        d_table_flags[idx] = 1;
        return;
    }

    // get previous item idx and table id
    uint32_t orig_idx_prev = sorted_item_indices[idx - 1];
    uint32_t table_id_prev = orig_idx_prev / n_samples;

    // if table id changed, mark new table
    d_table_flags[idx] = (table_id_curr != table_id_prev) ? 1 : 0;

    if (d_table_flags[idx] == 1) {
        // new table means new bucket
        d_bucket_flags[idx] = 1;
    } else {
        // check if signature changed from prev to curr
        uint32_t row_id_curr = orig_idx_curr % n_samples;
        uint32_t row_id_prev = orig_idx_prev % n_samples;
        const int8_t* sig_ptr_curr = X_sig +
                                     table_id_curr * (static_cast<size_t>(n_samples) * n_projections) +
                                     row_id_curr * n_projections;
        const int8_t* sig_ptr_prev = X_sig +
                                     table_id_prev * (static_cast<size_t>(n_samples) * n_projections) +
                                     row_id_prev * n_projections;
        // if signature changed, mark new bucket
        d_bucket_flags[idx] =
            are_signatures_equal(sig_ptr_curr, sig_ptr_prev, n_projections) ? 0 : 1;
    }
}

/**
 * @brief Scatter sorted data into final flat index
 */
__global__ void build_final_index_kernel(const int8_t* X_sig, const uint32_t* sorted_item_indices,
                                         const int* d_bucket_flags, const int* d_bucket_scan,
                                         int n_samples, int n_hash_tables, int n_projections,
                                         size_t n_items, int8_t* d_bucket_signatures,
                                         int* d_bucket_candidate_offsets, int* d_all_candidates) {

    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n_items)
        return;

    uint32_t orig_idx = sorted_item_indices[idx];
    uint32_t orig_row_id = orig_idx % n_samples;

    // each thread (each item) stores its original row idx to all_candidate_indices in sorted order
    d_all_candidates[idx] = orig_row_id;

    if (d_bucket_flags[idx] == 1) {
        // if this thread (item) marks start of a new bucket:
        // 1. write idx to bucket_candidate_offsets (store starting pos of this bucket's candidates)
        // 2. write signature for this bucket to d_bucket_signatures

        int bucket_idx = d_bucket_scan[idx];
        d_bucket_candidate_offsets[bucket_idx] = idx;

        // find src signature address:
        // (X_sig addr) + table_id * (size of table) + row_id * (size of signature)
        uint32_t table_id = orig_idx / n_samples;
        const int8_t* sig_src_ptr = X_sig +
                                    table_id * (static_cast<size_t>(n_samples) * n_projections) +
                                    orig_row_id * n_projections;
        // find dst signature address:
        // (d_bucket_signatures addr) + bucket_idx * (size of signature)
        int8_t* sig_dst_ptr = d_bucket_signatures + static_cast<size_t>(bucket_idx) * n_projections;

        for (int i = 0; i < n_projections; ++i) {
            sig_dst_ptr[i] = sig_src_ptr[i];
        }
    }
}

/**
 * @brief Build flat LSH index structure on GPU
 * @param[in] stream CUDA stream
 * @param[in] X_sig int8_t signature matrix (n_samples x n_hash_tables * n_projections)
 * @param[in] n_samples Number of input rows
 * @param[in] n_hash_tables Number of hash tables
 * @param[in] n_projections Number of projections
 * @return RPLSHIndex Index.
 */
RPLSHIndex build_index(cudaStream_t stream, const int8_t* X_sig, int n_samples, int n_hash_tables, int n_projections) {
    size_t n_items = static_cast<size_t>(n_samples) * n_hash_tables;
    dim3 block_size(256);
    dim3 grid_size((n_items + block_size.x - 1) / block_size.x);

    // allocate temp storage
    uint32_t *d_item_indices, *d_temp_indices;
    uint8_t *d_keys, *d_temp_keys;
    int *d_bucket_flags, *d_table_flags, *d_bucket_scan;
    CUDA_CHECK(cudaMalloc(&d_item_indices, n_items * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_temp_indices, n_items * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_keys, n_items * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_temp_keys, n_items * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_bucket_flags, n_items * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_table_flags, n_items * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_bucket_scan, n_items * sizeof(int)));

    // initialize sequence d_item_indices [0, 1, 2, ... n_items-1]
    thrust::sequence(thrust::cuda::par.on(stream), d_item_indices, d_item_indices + n_items, 0);

    // query memory requirements for radix sort
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_temp_keys,
                                    d_item_indices, d_temp_indices, n_items, 0, 8, stream);
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    // sort items lexicographically by signature, from least -> most significant signature byte
    for (int byte_idx = n_projections - 1; byte_idx >= 0; --byte_idx) {
        // extract nth byte for each item from corresponding signature in X_sig 
        extract_byte_key_kernel<<<grid_size, block_size, 0, stream>>>(
            X_sig, d_item_indices, n_samples, n_hash_tables, n_projections, byte_idx, d_keys);
        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_temp_keys,
                                        d_item_indices, d_temp_indices, n_items, 0, 8, stream);
        std::swap(d_item_indices, d_temp_indices);
    }

    // final sort by table id s.t. item groups are sorted by table:
    // d_item_indices = [table_0_item_0, table_0_item_1, ..., table_0_item_n-1,
    //                  table_1_item_0, table_1_item_1, ..., table_1_item_n-1,
    //                  ...]
    extract_table_id_key_kernel<<<grid_size, block_size, 0, stream>>>(d_item_indices, n_samples,
                                                                      n_items, d_keys);
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_temp_keys,
                                    d_item_indices, d_temp_indices, n_items, 0, 8, stream);
    std::swap(d_item_indices, d_temp_indices);

    // mark boundaries between buckets (groups of unique items) and tables
    mark_boundaries_kernel<<<grid_size, block_size, 0, stream>>>(
        X_sig, d_item_indices, n_samples, n_hash_tables, n_projections, n_items, d_bucket_flags,
        d_table_flags);

    // initialize index
    RPLSHIndex index;
    index.n_hash_tables = n_hash_tables;
    index.n_projections = n_projections;
    CUDA_CHECK(cudaMalloc(&index.table_bucket_offsets, (n_hash_tables + 1) * sizeof(int)));

    // run exclusive sum to get bucket offsets from binary flags, e.g.
    // d_bucket_flags = [1, 0, 0, 1, 0, 0, 1, 0, ...]
    // d_bucket_scan = [0, 1, 1, 2, 2, 2, 3, 3, ...]
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_bucket_flags, d_bucket_scan,
                                  n_items, stream);

    int* d_num_selected_out;
    CUDA_CHECK(cudaMalloc(&d_num_selected_out, sizeof(int)));

    // query memory requirements for select
    void* d_select_temp_storage = nullptr;
    size_t select_temp_storage_bytes = 0;
    cub::DeviceSelect::Flagged(d_select_temp_storage, select_temp_storage_bytes, d_bucket_scan,
                               d_table_flags, index.table_bucket_offsets, d_num_selected_out,
                               n_items, stream);
    CUDA_CHECK(cudaMalloc(&d_select_temp_storage, select_temp_storage_bytes));

    // select from d_bucket_scan using d_table_flags to get table offsets. e.g.
    // d_bucket_scan = [0, 1, 1, 2, 2, 2, 3, 3, ...]
    // d_table_flags = [1, 0, 0, 1, 0, 0, 1, 0, ...]
    // table_bucket_offsets = [0, 2, 3, ...]
    cub::DeviceSelect::Flagged(d_select_temp_storage, select_temp_storage_bytes, d_bucket_scan,
                               d_table_flags, index.table_bucket_offsets, d_num_selected_out,
                               n_items, stream);

    // get total number of unique buckets:
    // last_bucket_idx_val (buckets seen before last item) + last_bucket_flag_val (whether last bucket is new bucket)
    int last_bucket_idx_val, last_bucket_flag_val;
    CUDA_CHECK(cudaMemcpyAsync(&last_bucket_idx_val, d_bucket_scan + (n_items - 1), sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&last_bucket_flag_val, d_bucket_flags + (n_items - 1), sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    index.n_total_buckets = last_bucket_idx_val + last_bucket_flag_val;

    // allocate all_candidate_indices - original row idx for all items in sorted order
    CUDA_CHECK(cudaMalloc(&index.all_candidate_indices, n_items * sizeof(int)));
    // allocate bucket_candidate_offsets - start idx of each bucket's candidate indices in all_candidate_indices
    CUDA_CHECK(cudaMalloc(&index.bucket_candidate_offsets, (index.n_total_buckets + 1) * sizeof(int)));
    // allocate all_bucket_signatures - signature for all buckets in sorted order
    CUDA_CHECK(cudaMalloc(&index.all_bucket_signatures, static_cast<size_t>(index.n_total_buckets) *
                                                            n_projections * sizeof(int8_t)));

    // set terminating values in offset arrays
    int n_items_int = static_cast<int>(n_items);
    CUDA_CHECK(cudaMemcpyAsync(index.table_bucket_offsets + n_hash_tables, &index.n_total_buckets,
                               sizeof(int), cudaMemcpyHostToDevice, stream));  // for table_bucket_offsets, set total number of buckets
    CUDA_CHECK(cudaMemcpyAsync(index.bucket_candidate_offsets + index.n_total_buckets, &n_items_int,
                               sizeof(int), cudaMemcpyHostToDevice, stream));  // for bucket_candidate_offsets, set total number of items

    // scatter sorted data into final flat index
    build_final_index_kernel<<<grid_size, block_size, 0, stream>>>(
        X_sig, d_item_indices, d_bucket_flags, d_bucket_scan, n_samples, n_hash_tables,
        n_projections, n_items, index.all_bucket_signatures, index.bucket_candidate_offsets,
        index.all_candidate_indices);

    // cleanup
    CUDA_CHECK(cudaFree(d_temp_storage));
    CUDA_CHECK(cudaFree(d_item_indices));
    CUDA_CHECK(cudaFree(d_temp_indices));
    CUDA_CHECK(cudaFree(d_keys));
    CUDA_CHECK(cudaFree(d_temp_keys));
    CUDA_CHECK(cudaFree(d_bucket_flags));
    CUDA_CHECK(cudaFree(d_table_flags));
    CUDA_CHECK(cudaFree(d_bucket_scan));
    CUDA_CHECK(cudaFree(d_num_selected_out));
    CUDA_CHECK(cudaFree(d_select_temp_storage));

    return index;
}

} // namespace detail
} // namespace rplsh
} // namespace culsh
