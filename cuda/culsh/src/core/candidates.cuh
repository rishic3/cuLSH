#pragma once

#include "utils.cuh"
#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>

namespace culsh {
namespace core {

/**
 * @brief GPU candidates results
 */
struct Candidates {
    /**
     * @brief Device pointer to array of candidate indices for all queries.
     * Candidates for each query stored contiguously starting at query_candidate_offsets[i].
     * Size: [n_total_candidates]
     */
    int* query_candidate_indices = nullptr;

    /**
     * @brief Device pointer to array of number of candidates per query.
     * Size: [n_queries]
     */
    size_t* query_candidate_counts = nullptr;

    /**
     * @brief Start idx of each query's candidate indices in query_candidate_indices.
     * Size: [n_queries + 1]
     */
    size_t* query_candidate_offsets = nullptr;

    /**
     * @brief Metadata
     */
    int n_queries = 0;
    size_t n_total_candidates = 0;

    /**
     * @brief Default constructor
     */
    Candidates()
        : query_candidate_indices(nullptr), query_candidate_counts(nullptr),
          query_candidate_offsets(nullptr) {}

    /**
     * @brief Destructor
     */
    ~Candidates() { free(); }

    /**
     * @brief Move constructor
     */
    Candidates(Candidates&& other) noexcept
        : query_candidate_indices(other.query_candidate_indices),
          query_candidate_counts(other.query_candidate_counts),
          query_candidate_offsets(other.query_candidate_offsets), n_queries(other.n_queries),
          n_total_candidates(other.n_total_candidates) {

        // nullify moved-from object to prevent double-free
        other.query_candidate_indices = nullptr;
        other.query_candidate_counts = nullptr;
        other.query_candidate_offsets = nullptr;
        other.n_queries = 0;
        other.n_total_candidates = 0;
    }

    /**
     * @brief Move assignment operator
     */
    Candidates& operator=(Candidates&& other) noexcept {
        if (this != &other) {
            free();

            query_candidate_indices = other.query_candidate_indices;
            query_candidate_counts = other.query_candidate_counts;
            query_candidate_offsets = other.query_candidate_offsets;
            n_queries = other.n_queries;
            n_total_candidates = other.n_total_candidates;

            // nullify moved-from object to prevent double-free
            other.query_candidate_indices = nullptr;
            other.query_candidate_counts = nullptr;
            other.query_candidate_offsets = nullptr;
            other.n_queries = 0;
            other.n_total_candidates = 0;
        }
        return *this;
    }

    /**
     * @brief Delete copy constructor
     */
    Candidates(const Candidates&) = delete;

    /**
     * @brief Delete copy assignment operator
     */
    Candidates& operator=(const Candidates&) = delete;

    /**
     * @brief Check empty
     */
    bool empty() const {
        return query_candidate_indices == nullptr && query_candidate_counts == nullptr &&
               query_candidate_offsets == nullptr;
    }

    /**
     * @brief Free device memory
     */
    void free() {
        if (query_candidate_indices) {
            cudaFree(query_candidate_indices);
            query_candidate_indices = nullptr;
        }
        if (query_candidate_counts) {
            cudaFree(query_candidate_counts);
            query_candidate_counts = nullptr;
        }
        if (query_candidate_offsets) {
            cudaFree(query_candidate_offsets);
            query_candidate_offsets = nullptr;
        }
        n_queries = 0;
        n_total_candidates = 0;
    }

    /**
     * @brief Merge another Candidates object into this one
     *
     * @param[in] stream CUDA stream
     * @param[in] other Candidates to merge (will be freed)
     */
    void merge(cudaStream_t stream, Candidates&& other);
};

inline void Candidates::merge(cudaStream_t stream, Candidates&& other) {
    if (other.empty()) {
        return;
    }

    if (empty()) {
        // Move other into this
        *this = std::move(other);
        return;
    }

    // Compute new sizes
    int new_n_queries = n_queries + other.n_queries;
    size_t new_n_total = n_total_candidates + other.n_total_candidates;

    // Allocate new arrays
    int* new_indices = nullptr;
    size_t* new_counts = nullptr;
    size_t* new_offsets = nullptr;
    CUDA_CHECK_ALLOC(cudaMalloc(&new_indices, new_n_total * sizeof(int)));
    CUDA_CHECK_ALLOC(cudaMalloc(&new_counts, new_n_queries * sizeof(size_t)));
    CUDA_CHECK_ALLOC(cudaMalloc(&new_offsets, (new_n_queries + 1) * sizeof(size_t)));

    // Copy indices
    CUDA_CHECK(cudaMemcpyAsync(new_indices, query_candidate_indices,
                               n_total_candidates * sizeof(int), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(new_indices + n_total_candidates, other.query_candidate_indices,
                               other.n_total_candidates * sizeof(int), cudaMemcpyDeviceToDevice,
                               stream));

    // Copy counts
    CUDA_CHECK(cudaMemcpyAsync(new_counts, query_candidate_counts, n_queries * sizeof(size_t),
                               cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(new_counts + n_queries, other.query_candidate_counts,
                               other.n_queries * sizeof(size_t), cudaMemcpyDeviceToDevice, stream));

    // Copy this's offsets
    CUDA_CHECK(cudaMemcpyAsync(new_offsets, query_candidate_offsets,
                               (n_queries + 1) * sizeof(size_t), cudaMemcpyDeviceToDevice, stream));

    // Copy other's offsets and add base offset (n_total_candidates)
    // Ignore first offset since it is always 0
    thrust::transform(thrust::cuda::par.on(stream), other.query_candidate_offsets + 1,
                      other.query_candidate_offsets + other.n_queries + 1,
                      new_offsets + n_queries + 1, thrust::placeholders::_1 + n_total_candidates);

    cudaStreamSynchronize(stream);

    // Cleanup old objects
    free();
    other.free();

    // Update this object
    query_candidate_indices = new_indices;
    query_candidate_counts = new_counts;
    query_candidate_offsets = new_offsets;
    n_queries = new_n_queries;
    n_total_candidates = new_n_total;
}

} // namespace core
} // namespace culsh
