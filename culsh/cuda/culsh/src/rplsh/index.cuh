#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <stdexcept>

namespace culsh {
namespace rplsh {

/**
 * @brief GPU LSH index
 */
struct Index {
    /**
     * @brief Device pointer to flat sorted array of all candidate indices.
     * Candidates for each signature stored contiguously starting at signature_start_indices[i].
     * Size: [n_total_candidates]
     */
    int* all_candidate_indices = nullptr;

    /**
     * @brief Device pointer to flat sorted array of all bucket signatures for each hash tables.
     * Bucket signatures for each hash table stored contiguously starting at table_start_indices[i].
     * Size: [n_total_buckets * n_projections]
     */
    int8_t* all_bucket_signatures = nullptr;

    /**
     * @brief Start idx of each bucket's candidate indices in all_candidate_indices.
     * Size: [n_total_buckets + 1]
     */
    int* bucket_candidate_offsets = nullptr;

    /**
     * @brief Start idx of each table's signatures in all_bucket_signatures.
     * Size: [n_hash_tables + 1]
     */
    int* table_bucket_offsets = nullptr;

    /**
     * @brief Random projection matrix used to hash input.
     * Either float or double based on is_double flag.
     * Size: [n_hash_tables * n_projections * n_features]
     */
    void* P = nullptr;

    /**
     * @brief Metadata
     */
    int n_total_candidates = 0;
    int n_total_buckets = 0;
    int n_hash_tables = 0;
    int n_projections = 0;
    int n_features = 0;
    uint64_t seed = 0;
    bool is_double = false;

    /**
     * @brief Default constructor
     */
    Index()
        : all_candidate_indices(nullptr), all_bucket_signatures(nullptr),
          bucket_candidate_offsets(nullptr), table_bucket_offsets(nullptr), P(nullptr),
          n_total_candidates(0), n_total_buckets(0), n_hash_tables(0), n_projections(0),
          n_features(0), seed(0), is_double(false) {}

    /**
     * @brief Destructor
     */
    ~Index() { free(); }

    /**
     * @brief Move constructor
     */
    Index(Index&& other) noexcept
        : all_candidate_indices(other.all_candidate_indices),
          all_bucket_signatures(other.all_bucket_signatures),
          bucket_candidate_offsets(other.bucket_candidate_offsets),
          table_bucket_offsets(other.table_bucket_offsets), P(other.P),
          n_total_candidates(other.n_total_candidates), n_total_buckets(other.n_total_buckets),
          n_hash_tables(other.n_hash_tables), n_projections(other.n_projections),
          n_features(other.n_features), seed(other.seed), is_double(other.is_double) {

        // nullify moved-from object to prevent double-free
        other.all_candidate_indices = nullptr;
        other.all_bucket_signatures = nullptr;
        other.bucket_candidate_offsets = nullptr;
        other.table_bucket_offsets = nullptr;
        other.P = nullptr;
        other.n_total_candidates = 0;
        other.n_total_buckets = 0;
        other.n_hash_tables = 0;
        other.n_projections = 0;
        other.n_features = 0;
        other.seed = 0;
        other.is_double = false;
    }

    /**
     * @brief Move assignment operator
     */
    Index& operator=(Index&& other) noexcept {
        if (this != &other) {
            free();

            all_candidate_indices = other.all_candidate_indices;
            all_bucket_signatures = other.all_bucket_signatures;
            bucket_candidate_offsets = other.bucket_candidate_offsets;
            table_bucket_offsets = other.table_bucket_offsets;
            P = other.P;
            n_total_candidates = other.n_total_candidates;
            n_total_buckets = other.n_total_buckets;
            n_hash_tables = other.n_hash_tables;
            n_projections = other.n_projections;
            n_features = other.n_features;
            seed = other.seed;
            is_double = other.is_double;

            // nullify moved-from object to prevent double-free
            other.all_candidate_indices = nullptr;
            other.all_bucket_signatures = nullptr;
            other.bucket_candidate_offsets = nullptr;
            other.table_bucket_offsets = nullptr;
            other.P = nullptr;
            other.n_total_candidates = 0;
            other.n_total_buckets = 0;
            other.n_hash_tables = 0;
            other.n_projections = 0;
            other.n_features = 0;
            other.seed = 0;
            other.is_double = false;
        }
        return *this;
    }

    /**
     * @brief Delete copy constructor
     */
    Index(const Index&) = delete;

    /**
     * @brief Delete copy assignment operator
     */
    Index& operator=(const Index&) = delete;

    /**
     * @brief Check empty
     */
    bool empty() const {
        return all_candidate_indices == nullptr && all_bucket_signatures == nullptr &&
               bucket_candidate_offsets == nullptr && table_bucket_offsets == nullptr &&
               P == nullptr;
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
        // projection matrix P
        size_t P_elements = static_cast<size_t>(n_hash_tables) * n_projections * n_features;
        total_size_bytes += P_elements * (is_double ? sizeof(double) : sizeof(float));

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
        if (P) {
            cudaFree(P);
            P = nullptr;
        }
        n_total_candidates = 0;
        n_total_buckets = 0;
        n_hash_tables = 0;
        n_projections = 0;
        n_features = 0;
        seed = 0;
        is_double = false;
    }

    /**
     * @brief Get projection matrix as float pointer
     * @throws std::invalid_argument if is_double
     */
    float* P_float() const {
        if (is_double) {
            throw std::invalid_argument("Projection matrix is not float precision");
        }
        return static_cast<float*>(P);
    }

    /**
     * @brief Get projection matrix as double pointer
     * @throws std::invalid_argument if !is_double
     */
    double* P_double() const {
        if (!is_double) {
            throw std::invalid_argument("Projection matrix is not double precision");
        }
        return static_cast<double*>(P);
    }
};

} // namespace rplsh
} // namespace culsh
