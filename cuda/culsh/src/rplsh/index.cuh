#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace culsh {
namespace rplsh {

/**
 * @brief Managed GPU LSH index.
 */
class RPLSHIndex {
public:
    /**
     * @brief Device pointer to flat sorted array of all candidate indices.
     * @note Candidates for each signature stored contiguously starting at signature_start_indices[i].
     */
    int* all_candidate_indices;

    /**
     * @brief Device pointer to flat sorted array of all bucket signatures for each hash tables.
     * @note Bucket signatures for each hash table stored contiguously starting at table_start_indices[i].
     */
    int8_t* all_bucket_signatures;

    /**
     * @brief Start idx of each bucket's candidate indices in all_candidate_indices.
     */
    int* bucket_candidate_offsets;

    /**
     * @brief Start idx of each table's signatures in all_bucket_signatures.
     */
    int* table_bucket_offsets;

    /**
     * @brief Metadata
     */
    int n_total_buckets;
    int n_hash_tables;
    int n_projections;

    /**
     * @brief Default constructor
     * 
     */
    RPLSHIndex() : 
        all_candidate_indices(nullptr),
        all_bucket_signatures(nullptr), 
        bucket_candidate_offsets(nullptr),
        table_bucket_offsets(nullptr),
        n_total_buckets(0),
        n_hash_tables(0),
        n_projections(0) {}

    /**
     * @brief Destructor
     */
    ~RPLSHIndex() {
        free_device_memory();
    }

    /**
     * @brief Move constructor
     */
    RPLSHIndex(RPLSHIndex&& other) noexcept :
        all_candidate_indices(other.all_candidate_indices),
        all_bucket_signatures(other.all_bucket_signatures),
        bucket_candidate_offsets(other.bucket_candidate_offsets),
        table_bucket_offsets(other.table_bucket_offsets),
        n_total_buckets(other.n_total_buckets),
        n_hash_tables(other.n_hash_tables),
        n_projections(other.n_projections) {
        
        // nullify moved-from object to prevent double-free
        other.all_candidate_indices = nullptr;
        other.all_bucket_signatures = nullptr;
        other.bucket_candidate_offsets = nullptr;
        other.table_bucket_offsets = nullptr;
        other.n_total_buckets = 0;
        other.n_hash_tables = 0;
        other.n_projections = 0;
    }

    /**
     * @brief Move assignment operator
     */
    RPLSHIndex& operator=(RPLSHIndex&& other) noexcept {
        if (this != &other) {
            free_device_memory();

            all_candidate_indices = other.all_candidate_indices;
            all_bucket_signatures = other.all_bucket_signatures;
            bucket_candidate_offsets = other.bucket_candidate_offsets;
            table_bucket_offsets = other.table_bucket_offsets;
            n_total_buckets = other.n_total_buckets;
            n_hash_tables = other.n_hash_tables;
            n_projections = other.n_projections;

            // nullify moved-from object to prevent double-free
            other.all_candidate_indices = nullptr;
            other.all_bucket_signatures = nullptr;
            other.bucket_candidate_offsets = nullptr;
            other.table_bucket_offsets = nullptr;
            other.n_total_buckets = 0;
            other.n_hash_tables = 0;
            other.n_projections = 0;
        }
        return *this;
    }

    /**
     * @brief Delete copy constructor
     */
    RPLSHIndex(const RPLSHIndex&) = delete;

    /**
     * @brief Delete copy assignment operator
     */
    RPLSHIndex& operator=(const RPLSHIndex&) = delete;

    /**
     * @brief Check empty
     */
    bool empty() const {
        return all_candidate_indices == nullptr && 
               all_bucket_signatures == nullptr && 
               bucket_candidate_offsets == nullptr && 
               table_bucket_offsets == nullptr;
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
        if (all_bucket_signatures != nullptr) {
            cudaFree(all_bucket_signatures);
            all_bucket_signatures = nullptr;
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
