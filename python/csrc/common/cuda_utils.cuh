#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace culsh {
namespace python {

#define CUDA_CHECK_THROW(call)                                                                     \
    do {                                                                                           \
        cudaError_t err = call;                                                                    \
        if (err != cudaSuccess) {                                                                  \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));       \
        }                                                                                          \
    } while (0)

#define CUBLAS_CHECK_THROW(call)                                                                   \
    do {                                                                                           \
        cublasStatus_t status = call;                                                              \
        if (status != CUBLAS_STATUS_SUCCESS) {                                                     \
            throw std::runtime_error("cuBLAS error: " + std::to_string(static_cast<int>(status))); \
        }                                                                                          \
    } while (0)

/**
 * @brief Manage CUDA resources (cuBLAS handle, stream)
 */
class CUDAResourceManager {
protected:
    cublasHandle_t cublas_handle_ = nullptr;
    cudaStream_t stream_ = nullptr;

public:
    CUDAResourceManager() {
        CUBLAS_CHECK_THROW(cublasCreate(&cublas_handle_));
        CUDA_CHECK_THROW(cudaStreamCreate(&stream_));
    }

    virtual ~CUDAResourceManager() {
        if (stream_) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
        if (cublas_handle_) {
            cublasDestroy(cublas_handle_);
            cublas_handle_ = nullptr;
        }
    }

    // Non-copyable
    CUDAResourceManager(const CUDAResourceManager&) = delete;
    CUDAResourceManager& operator=(const CUDAResourceManager&) = delete;

    // Accessors
    cublasHandle_t cublas_handle() const { return cublas_handle_; }
    cudaStream_t stream() const { return stream_; }

    void synchronize() { CUDA_CHECK_THROW(cudaStreamSynchronize(stream_)); }
};

} // namespace python
} // namespace culsh
