#pragma once

#include "../core/index.cuh"
#include <cstdint>
#include <cuda_runtime.h>

namespace culsh {
namespace minhash {

/**
 * @brief MinHash index. Owns core::Index plus hash integers A, B and metadata.
 */
struct Index {
    core::Index core;

    uint32_t* A = nullptr;
    uint32_t* B = nullptr;

    Index() = default;
    ~Index() { free(); }

    Index(Index&& other) noexcept : core(std::move(other.core)), A(other.A), B(other.B) {
        other.A = nullptr;
        other.B = nullptr;
    }

    Index& operator=(Index&& other) noexcept {
        if (this != &other) {
            free();
            core = std::move(other.core);
            A = other.A;
            B = other.B;

            other.A = nullptr;
            other.B = nullptr;
        }
        return *this;
    }

    Index(const Index&) = delete;
    Index& operator=(const Index&) = delete;

    bool empty() const { return core.empty() && A == nullptr && B == nullptr; }

    size_t size_bytes() const {
        size_t total = core.size_bytes();
        if (A != nullptr) {
            total += static_cast<size_t>(core.n_hash_tables) * core.n_hashes * sizeof(uint32_t);
        }
        if (B != nullptr) {
            total += static_cast<size_t>(core.n_hash_tables) * core.n_hashes * sizeof(uint32_t);
        }
        return total;
    }

    void free() {
        core.free();
        if (A) {
            cudaFree(A);
            A = nullptr;
        }
        if (B) {
            cudaFree(B);
            B = nullptr;
        }
    }
};

} // namespace minhash
} // namespace culsh
