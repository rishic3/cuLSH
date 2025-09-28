#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace culsh {
namespace rplsh {

/**
 * @brief Managed candidates results.
 */
class Candidates {
public:
    /**
     * @brief Device pointer to array of candidate indices for all queries.
     */
    int* query_candidate_indices;

    /**
     * @brief Device pointer to array of number of candidates per query.
     */
    size_t* query_candidate_counts;

    /**
     * @brief Start idx of each query's candidate indices in query_candidate_indices.
     */
    size_t* query_candidate_offsets;

    /**
     * @brief Default constructor
     */
    Candidates()
        : query_candidate_indices(nullptr), query_candidate_counts(nullptr),
          query_candidate_offsets(nullptr) {}

    /**
     * @brief Destructor
     */
    ~Candidates() { free_device_memory(); }

    /**
     * @brief Move constructor
     */
    Candidates(Candidates&& other) noexcept
        : query_candidate_indices(other.query_candidate_indices),
          query_candidate_counts(other.query_candidate_counts),
          query_candidate_offsets(other.query_candidate_offsets) {

        // nullify moved-from object to prevent double-free
        other.query_candidate_indices = nullptr;
        other.query_candidate_counts = nullptr;
        other.query_candidate_offsets = nullptr;
    }

    /**
     * @brief Move assignment operator
     */
    Candidates& operator=(Candidates&& other) noexcept {
        if (this != &other) {
            free_device_memory();

            query_candidate_indices = other.query_candidate_indices;
            query_candidate_counts = other.query_candidate_counts;
            query_candidate_offsets = other.query_candidate_offsets;

            // nullify moved-from object to prevent double-free
            other.query_candidate_indices = nullptr;
            other.query_candidate_counts = nullptr;
            other.query_candidate_offsets = nullptr;
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

private:
    /**
     * @brief Free device memory
     */
    void free_device_memory() {
        if (query_candidate_indices != nullptr) {
            cudaFree(query_candidate_indices);
            query_candidate_indices = nullptr;
        }
        if (query_candidate_counts != nullptr) {
            cudaFree(query_candidate_counts);
            query_candidate_counts = nullptr;
        }
        if (query_candidate_offsets != nullptr) {
            cudaFree(query_candidate_offsets);
            query_candidate_offsets = nullptr;
        }
    }
};

} // namespace rplsh
} // namespace culsh
