#pragma once

#include "kernels/hash.cuh"
#include "kernels/indexing.cuh"
#include "kernels/projections.cuh"
#include <cuda_runtime.h>
#include <culsh/rplsh/params.hpp>
#include <culsh/rplsh/rplsh.hpp>
#include <curand.h>

namespace culsh {
namespace rplsh {

/**
 * @brief GPU LSH index
 */
struct RPLSHIndex {
    /**
     * Device pointer to flat sorted array of unique hash table signatures.
     * Hash table signatures stored contiguously starting at table_start_indices[i].
     */
    int8_t* unique_signatures;

    /**
     * Device pointer to flat array of all candidate indices.
     * Candidates for each signature stored contiguously starting at signature_start_indices[i].
     */
    int* candidate_indices;

    /**
     * Start idx of each hash table in unique_signatures.
     */
    int* table_start_indices;

    /**
     * Number of unique signatures for each hash table.
     */
    int* table_signature_counts;

    /**
     * Start idx of each unique signature in candidate_indices.
     */
    int* signature_start_indices;

    /**
     * Number of candidates for each unique signature.
     */
    int* signature_sizes;

    /**
     * Metadata
     */
    int n_fit_rows;
    int total_unique_signatures;
    int total_candidate_indices;
};

} // namespace rplsh
} // namespace culsh
