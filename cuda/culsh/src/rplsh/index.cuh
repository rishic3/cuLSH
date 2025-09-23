#pragma once

#include <cstdint>

namespace culsh {
namespace rplsh {

/**
 * @brief GPU LSH index
 */
struct RPLSHIndex {
    /**
     * Device pointer to flat sorted array of all candidate indices.
     * Candidates for each signature stored contiguously starting at signature_start_indices[i].
     */
    int* all_candidate_indices;

    /**
     * Device pointer to flat sorted array of all bucket signatures for each hash tables.
     * Bucket signatures for each hash table stored contiguously starting at table_start_indices[i].
     */
    int8_t* all_bucket_signatures;

    /**
     * Start idx of each bucket's candidate indices in all_candidate_indices.
     */
    int* bucket_candidate_offsets;

    /**
     * Start idx of each table's signatures in all_bucket_signatures.
     */
    int* table_bucket_offsets;

    /**
     * Metadata
     */
    int n_total_buckets;
    int n_hash_tables;
    int n_projections;
};

} // namespace rplsh
} // namespace culsh
