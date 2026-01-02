#pragma once

#include "../core/index.cuh"
#include <cuda_runtime.h>

namespace culsh {
namespace pslsh {

/**
 * @brief p-Stable LSH index. Owns core::Index plus projection vector/offset and metadata.
 */
struct Index {
    core::Index core;

    void* P = nullptr;
    void* b = nullptr;
    bool is_double = false;

    Index() = default;
    ~Index() { free(); }

    Index(Index&& other) noexcept
        : core(std::move(other.core)), P(other.P), b(other.b), is_double(other.is_double) {
        other.P = nullptr;
        other.b = nullptr;
        other.is_double = false;
    }

    Index& operator=(Index&& other) noexcept {
        if (this != &other) {
            free();
            core = std::move(other.core);
            P = other.P;
            b = other.b;
            is_double = other.is_double;

            other.P = nullptr;
            other.b = nullptr;
            other.is_double = false;
        }
        return *this;
    }

    Index(const Index&) = delete;
    Index& operator=(const Index&) = delete;

    bool empty() const { return core.empty() && P == nullptr && b == nullptr; }

    size_t size_bytes() const {
        size_t total = core.size_bytes();
        if (P != nullptr && core.n_features > 0) {
            size_t P_elements = static_cast<size_t>(core.n_hash_tables) * core.n_hashes *
                                static_cast<size_t>(core.n_features);
            total += P_elements * (is_double ? sizeof(double) : sizeof(float));
        }
        if (b != nullptr) {
            size_t b_elements = static_cast<size_t>(core.n_hash_tables) * core.n_hashes;
            total += b_elements * (is_double ? sizeof(double) : sizeof(float));
        }
        return total;
    }

    void free() {
        core.free();
        if (P) {
            cudaFree(P);
            P = nullptr;
        }
        if (b) {
            cudaFree(b);
            b = nullptr;
        }
        is_double = false;
    }

    const float* P_float() const { return static_cast<const float*>(P); }
    const double* P_double() const { return static_cast<const double*>(P); }
    const float* b_float() const { return static_cast<const float*>(b); }
    const double* b_double() const { return static_cast<const double*>(b); }
};

} // namespace pslsh
} // namespace culsh
