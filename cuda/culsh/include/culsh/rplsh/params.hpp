#pragma once

#include <cstdint>

namespace culsh {

/**
 * @brief Random Projection LSH parameters
 */
struct RPLSHParams {
    /**
     * @brief The number of hash tables. This parameter corresponds to an OR-amplification of
     * the locality-sensitive family. A higher value increases the probability of finding
     * a candidate neighbor. Corresponds to 'b' in the amplified probability (1 - (1 - p^r)^b).
     */
    int n_hash_tables;

    /**
     * @brief The number of random projections per hash table. This parameter corresponds to an
     * AND-amplification of the locality-sensitive family. A higher value decreases the
     * probability of finding a candidate neighbor. Corresponds to 'r' in the amplified
     * probability (1 - (1 - p^r)^b).
     */
    int n_projections;

    /**
     * @brief Optional seed used to generate random projections.
     */
    uint64_t seed;
};

} // namespace culsh
