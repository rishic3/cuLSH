#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace culsh {
namespace rplsh {

/**
 * @brief GPU candidates results
 */
struct Candidates {
    /**
     * @brief Device pointer to array of candidate indices for all queries.
     */
    int* query_candidate_indices = nullptr;

    /**
     * @brief Device pointer to array of number of candidates per query.
     */
    size_t* query_candidate_counts = nullptr;

    /**
     * @brief Start idx of each query's candidate indices in query_candidate_indices.
     */
    size_t* query_candidate_offsets = nullptr;

    /**
     * @brief Metadata
     */
    int n_queries = 0;
    size_t n_total_candidates = 0;

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
};

} // namespace rplsh
} // namespace culsh
