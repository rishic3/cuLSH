#pragma once

#include <cstdint>

namespace culsh {

/**
 * @brief p-Stable LSH parameters
 */
struct PStableLSHParams {
    /**
     * @brief The number of hash tables. This parameter corresponds to an OR-amplification of
     * the locality-sensitive family. A higher value increases the probability of finding
     * a candidate neighbor. Corresponds to 'b' in the amplified probability (1 - (1 - p_w^r)^b),
     * where p_w is the probability of collision for window size w.
     */
    int n_hash_tables;

    /**
     * @brief The number of hash functions per hash table. This parameter corresponds
     * to an AND-amplification of the locality-sensitive family. A higher value decreases the
     * probability of finding a candidate neighbor. Corresponds to 'r' in the amplified
     * probability (1 - (1 - p_w^r)^b), where p_w is the probability of collision for window size w.
     */
    int n_hashes;

    /**
     * @brief The quantization width for projections. Determines the resolution of the hash function
     * by defining the physical size of the hash buckets. This controls the 'slope' of the
     * probability curve relative to distance: larger window size increases the base collision
     * probability.
     */
    int window_size;

    /**
     * @brief Optional seed used to generate random hash functions.
     */
    uint64_t seed;
};

} // namespace culsh
