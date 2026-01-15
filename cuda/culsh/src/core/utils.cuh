#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>

// Always check cudaMalloc calls. These are synchronous anyway so it's fine.
#define CUDA_CHECK_ALLOC(call)                                                                     \
    do {                                                                                           \
        cudaError_t error = call;                                                                  \
        if (error != cudaSuccess) {                                                                \
            fprintf(stderr, "CUDA allocation error at %s:%d - %s\n", __FILE__, __LINE__,           \
                    cudaGetErrorString(error));                                                    \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

#ifndef NDEBUG
#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t error = call;                                                                  \
        if (error != cudaSuccess) {                                                                \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__,                      \
                    cudaGetErrorString(error));                                                    \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

#define CURAND_CHECK(call)                                                                         \
    do {                                                                                           \
        curandStatus_t status = call;                                                              \
        if (status != CURAND_STATUS_SUCCESS) {                                                     \
            fprintf(stderr, "cuRAND error at %s:%d - %d\n", __FILE__, __LINE__, status);           \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

#define CUBLAS_CHECK(call)                                                                         \
    do {                                                                                           \
        cublasStatus_t status = call;                                                              \
        if (status != CUBLAS_STATUS_SUCCESS) {                                                     \
            fprintf(stderr, "cuBLAS error at %s:%d - %d\n", __FILE__, __LINE__, status);           \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)
#else
#define CUDA_CHECK(call) call
#define CURAND_CHECK(call) call
#define CUBLAS_CHECK(call) call
#endif

namespace culsh {
namespace core {
namespace detail {

/**
 * @brief Ensures sufficient temporary storage is allocated for a CUB operations
 *
 * Checks if the current allocation size (current_bytes) is sufficient for the
 * requested size (required_bytes). If not free the existing memory and
 * update the d_temp_storage allocation to required_bytes.
 *
 * @param[in,out] d_temp_storage Pointer to device memory pointer
 * @param[in,out] current_bytes Reference to current size in bytes
 * @param[in] required_bytes Size in bytes required for the next operation
 */
inline void ensure_temp_storage(void** d_temp_storage, size_t& current_bytes,
                                size_t required_bytes) {
    if (required_bytes > current_bytes) {
        if (*d_temp_storage != nullptr) {
            CUDA_CHECK(cudaFree(*d_temp_storage));
        }
        CUDA_CHECK_ALLOC(cudaMalloc(d_temp_storage, required_bytes));
        current_bytes = required_bytes;
    }
}

} // namespace detail
} // namespace core
} // namespace culsh
