#pragma once

#include "candidates.cuh"
#include "constants.cuh"
#include "index.cuh"
#include "utils.cuh"
#include <cassert>
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
namespace core {
namespace detail {

/**
 * @brief Compute distance between each query and its matched candidates.
 * @param[in] X Device pointer to input matrix (n_samples x n_features)
 * @param[in] Q Device pointer to query matrix (n_queries x n_features)
 * @param[in] candidate_indices Device pointer to array of candidate indices for each query
 * @param[in] candidate_offsets Device pointer to array of offsets for each query's candidates
 * @param[in] n_features Number of features
 * @param[in] dist_fn Distance function
 * @param[out] out_distances Device pointer to array of output distances
 */
template <typename DType, typename DistFunc>
__global__ void compute_distances_kernel(const DType* X, const DType* Q, const int* candidate_indices,
                                         const size_t* candidate_offsets, int n_features, DistFunc dist_fn,
                                         DType* out_distances) {
    int query_idx = blockIdx.x;
    size_t start = candidate_offsets[query_idx];
    size_t end = candidate_offsets[query_idx + 1];

    const DType* query = Q + query_idx * n_features;

    for (size_t c = start + threadIdx.x; c < end; c += core::BLOCK_SIZE) {
        int cand_idx = candidate_indices[c];
        const DType* sample = X + cand_idx * n_features;
        out_distances[c] = dist_fn(query, sample, n_features);
    }
}

/**
 * @brief Refine candidates to top-k based on ground-truth distance
 * @param[in] X Device pointer to input matrix (n_samples x n_features)
 * @param[in] Q Device pointer to query matrix (n_queries x n_features)
 * @param[in] candidates Device candidates
 * @param[in] n_samples Number of rows in X
 * @param[in] n_queries Number of rows in Q
 * @param[in] n_features Dimensionality of each input sample
 * @param[in] dist_fn Distance function
 * @return RefinedCandidates object
 */
template <typename DType, typename DistFunc>
RefinedCandidates refine_candidates(cudaStream_t stream, const DType* X, const DType* Q, const Candidates* candidates,
                                    int n_features, DistFunc dist_fn, int k) {
    // TODO
}

} // namespace detail
} // namespace core
} // namespace culsh
