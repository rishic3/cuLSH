#pragma once

#include "candidates.cuh"
#include "fit.cuh"
#include "index.cuh"
#include "query.cuh"
#include "utils.cuh"
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
 * @brief Fit and query to get all-neighbors candidates
 * @param[in] stream CUDA stream
 * @param[in] X_sig Device pointer to signature matrix (n_samples x n_hash_tables * n_hashes)
 * @param[in] n_samples Number of input rows
 * @param[in] n_hash_tables Number of hash tables
 * @param[in] n_hashes Number of hashes per table
 * @return Candidates object
 */
Candidates fit_query(cudaStream_t stream, const int8_t* X_sig, int n_samples, int n_hash_tables,
                     int n_hashes) {
    size_t n_items = static_cast<size_t>(n_samples) * n_hash_tables;

    // Fit and query on the input X_sig to get all-neighbors candidates (i.e. X_sig == Q_sig).
    // Build the index and also fill a d_item_to_bucket mapping of (row,table) item -> bucket_id.
    // This is analogous to matched_bucket_indices in standalone query (no need to search
    // for matching buckets - every row has a signature and belongs to one bucket per table as
    // given by the index). Querying is the same as standalone query, proceeding directly to
    // query_from_matched_buckets.

    int* d_item_to_bucket;
    CUDA_CHECK(cudaMalloc(&d_item_to_bucket, n_items * sizeof(int)));

    // Build the index and fill d_item_to_bucket during final scatter
    Index index = fit_index(stream, X_sig, n_samples, n_hash_tables, n_hashes, d_item_to_bucket);

    // Query using precomputed bucket IDs
    Candidates candidates =
        query_from_matched_buckets(stream, d_item_to_bucket, n_samples, n_hash_tables, &index);

    CUDA_CHECK(cudaFree(d_item_to_bucket));
    index.free();

    return candidates;
}

} // namespace detail
} // namespace rplsh
} // namespace culsh
