#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace culsh {
namespace rplsh {

/**
 * @brief GPU LSH index with RAII semantics
 *
 * Key format: (table_id << n_projections) | packed_signature
 */
class Index {
public:
    /**
     * @brief Device pointer to flat sorted array of all candidate indices.
     * Candidates for each bucket stored contiguously starting at bucket_candidate_offsets[i].
     */
    int* all_candidate_indices;

    /**
     * @brief Device pointer to sorted array of packed bucket keys.
     * Each key encodes (table_id << n_projections) | signature_bits.
     */
    uint32_t* bucket_keys;

    /**
     * @brief Start idx of each bucket's candidate indices in all_candidate_indices.
     */
    int* bucket_candidate_offsets;

    /**
     * @brief Start idx of each table's buckets in bucket_keys.
     */
    int* table_bucket_offsets;

    /**
     * @brief Metadata
     */
    int n_total_candidates;
    int n_total_buckets;
    int n_hash_tables;
    int n_projections;

    /**
     * @brief Default constructor
     */
    Index()
        : all_candidate_indices(nullptr), bucket_keys(nullptr), bucket_candidate_offsets(nullptr),
          table_bucket_offsets(nullptr), n_total_candidates(0), n_total_buckets(0),
          n_hash_tables(0), n_projections(0) {}

    /**
     * @brief Destructor
     */
    ~Index() { free_device_memory(); }

    /**
     * @brief Move constructor
     */
    Index(Index&& other) noexcept
        : all_candidate_indices(other.all_candidate_indices), bucket_keys(other.bucket_keys),
          bucket_candidate_offsets(other.bucket_candidate_offsets),
          table_bucket_offsets(other.table_bucket_offsets),
          n_total_candidates(other.n_total_candidates), n_total_buckets(other.n_total_buckets),
          n_hash_tables(other.n_hash_tables), n_projections(other.n_projections) {

        // nullify moved-from object to prevent double-free
        other.all_candidate_indices = nullptr;
        other.bucket_keys = nullptr;
        other.bucket_candidate_offsets = nullptr;
        other.table_bucket_offsets = nullptr;
        other.n_total_candidates = 0;
        other.n_total_buckets = 0;
        other.n_hash_tables = 0;
        other.n_projections = 0;
    }

    /**
     * @brief Move assignment operator
     */
    Index& operator=(Index&& other) noexcept {
        if (this != &other) {
            free_device_memory();

            all_candidate_indices = other.all_candidate_indices;
            bucket_keys = other.bucket_keys;
            bucket_candidate_offsets = other.bucket_candidate_offsets;
            table_bucket_offsets = other.table_bucket_offsets;
            n_total_candidates = other.n_total_candidates;
            n_total_buckets = other.n_total_buckets;
            n_hash_tables = other.n_hash_tables;
            n_projections = other.n_projections;

            // nullify moved-from object to prevent double-free
            other.all_candidate_indices = nullptr;
            other.bucket_keys = nullptr;
            other.bucket_candidate_offsets = nullptr;
            other.table_bucket_offsets = nullptr;
            other.n_total_candidates = 0;
            other.n_total_buckets = 0;
            other.n_hash_tables = 0;
            other.n_projections = 0;
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
        return all_candidate_indices == nullptr && bucket_keys == nullptr &&
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
        // bucket_keys
        total_size_bytes += n_total_buckets * sizeof(uint32_t);
        // all_candidate_indices
        total_size_bytes += n_total_candidates * sizeof(int);

        return total_size_bytes;
    }

private:
    /**
     * @brief Free device memory
     */
    void free_device_memory() {
        if (all_candidate_indices != nullptr) {
            cudaFree(all_candidate_indices);
            all_candidate_indices = nullptr;
        }
        if (bucket_keys != nullptr) {
            cudaFree(bucket_keys);
            bucket_keys = nullptr;
        }
        if (bucket_candidate_offsets != nullptr) {
            cudaFree(bucket_candidate_offsets);
            bucket_candidate_offsets = nullptr;
        }
        if (table_bucket_offsets != nullptr) {
            cudaFree(table_bucket_offsets);
            table_bucket_offsets = nullptr;
        }
    }
};

} // namespace rplsh
} // namespace culsh
