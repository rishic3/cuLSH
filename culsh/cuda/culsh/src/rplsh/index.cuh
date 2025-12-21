#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace culsh {
namespace rplsh {

/**
 * @brief GPU LSH index
 */
struct Index {
    /**
     * @brief Device pointer to flat sorted array of all candidate indices.
     * Candidates for each signature stored contiguously starting at signature_start_indices[i].
     */
    int* all_candidate_indices = nullptr;

    /**
     * @brief Device pointer to flat sorted array of all bucket signatures for each hash tables.
     * Bucket signatures for each hash table stored contiguously starting at table_start_indices[i].
     */
    int8_t* all_bucket_signatures = nullptr;

    /**
     * @brief Start idx of each bucket's candidate indices in all_candidate_indices.
     */
    int* bucket_candidate_offsets = nullptr;

    /**
     * @brief Start idx of each table's signatures in all_bucket_signatures.
     */
    int* table_bucket_offsets = nullptr;

    /**
     * @brief Metadata
     */
    int n_total_candidates = 0;
    int n_total_buckets = 0;
    int n_hash_tables = 0;
    int n_projections = 0;

    /**
     * @brief Check empty
     */
    bool empty() const {
        return all_candidate_indices == nullptr && all_bucket_signatures == nullptr &&
               bucket_candidate_offsets == nullptr && table_bucket_offsets == nullptr;
    }

    /**
     * @brief Compute total device memory size of index
     */
    size_t device_size() const {
        if (empty()) {
            return 0;
        }

        size_t total_size_bytes = 0;
        // bucket_candidate_offsets
        total_size_bytes += (n_total_buckets + 1) * sizeof(int);
        // table_bucket_offsets
        total_size_bytes += (n_hash_tables + 1) * sizeof(int);
        // all_bucket_signatures
        total_size_bytes += n_total_buckets * n_projections * sizeof(int8_t);
        // all_candidate_indices
        total_size_bytes += n_total_candidates * sizeof(int);

        return total_size_bytes;
    }

    /**
     * @brief Free device memory
     */
    void free() {
        if (all_candidate_indices) {
            cudaFree(all_candidate_indices);
            all_candidate_indices = nullptr;
        }
        if (all_bucket_signatures) {
            cudaFree(all_bucket_signatures);
            all_bucket_signatures = nullptr;
        }
        if (bucket_candidate_offsets) {
            cudaFree(bucket_candidate_offsets);
            bucket_candidate_offsets = nullptr;
        }
        if (table_bucket_offsets) {
            cudaFree(table_bucket_offsets);
            table_bucket_offsets = nullptr;
        }
        n_total_candidates = 0;
        n_total_buckets = 0;
        n_hash_tables = 0;
        n_projections = 0;
    }
};

} // namespace rplsh
} // namespace culsh
