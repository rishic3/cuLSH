#pragma once

#include "../utils/utils.cuh"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdexcept>

namespace culsh {
namespace rplsh {
namespace detail {

template <typename DType>
__global__ void compute_signatures_kernel(const DType* X_hash, int n_rows, int n_hash_tables,
                                          int n_projections, int8_t* X_signatures) {

    // each thread computes one signature
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total_elements = static_cast<size_t>(n_rows) * n_hash_tables * n_projections;

    if (idx >= total_elements)
        return;

    size_t proj_idx = idx % n_projections;
    size_t table_idx = (idx / n_projections) % n_hash_tables;
    size_t row_idx = idx / (n_projections * n_hash_tables);

    // index of input hash
    size_t hash_idx =
        row_idx * (n_hash_tables * n_projections) + table_idx * n_projections + proj_idx;

    // index of output signature
    size_t sig_idx = table_idx * (n_rows * n_projections) + row_idx * n_projections + proj_idx;

    // convert to binary signature
    X_signatures[sig_idx] = (X_hash[hash_idx] > DType(0)) ? int8_t(1) : int8_t(0);
}

template <typename DType>
void compute_signatures(cudaStream_t stream, const DType* X_hash, int n_rows, int n_hash_tables,
                        int n_projections, int8_t* X_signatures) {
    dim3 block_size(256);
    dim3 grid_size((n_rows + block_size.x - 1) / block_size.x);
    compute_signatures_kernel<<<grid_size, block_size, 0, stream>>>(X_hash, n_rows, n_hash_tables,
                                                                    n_projections, X_signatures);
}

/**
 * @brief Hash the input vectors X using random projection matrix P.
 * @param[in] cublas_handle cuBLAS handle.
 * @param[in] stream CUDA stream.
 * @param[in] X Input vectors (n_rows x n_cols).
 * @param[in] P Random projection matrix (n_hash x n_cols).
 * @param[in] n_rows Number of input vectors.
 * @param[in] n_cols Dimensionality of each input vector.
 * @param[in] n_hash Total projection vectors (n_hash_tables * n_projections).
 * @param[out] X_hash Hashed input vectors (n_rows x n_hash).
 */
template <typename DType>
void hash(cublasHandle_t cublas_handle, cudaStream_t stream, const DType* X, const DType* P,
          int n_rows, int n_cols, int n_hash, DType* X_hash) {

    cublasSetStream(cublas_handle, stream);

    const DType alpha = DType(1.0);
    const DType beta = DType(0.0);

    // We want X * P^T = X_hash in row-major world.
    // In cuBLAS's column-major world (denoting it _c), we have X_c = X^T, P_c = P^T.
    // We can compute X_c^T * P_c = X_hash_c = X_hash^T.
    // To get X_hash in row-major, we need X_hash_c^T = (X_c^T * P_c)^T = P_c^T * X_c
    if constexpr (std::is_same_v<DType, float>) {
        CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, n_hash, n_rows, n_cols,
                                 &alpha, P, n_cols, X, n_cols, &beta, X_hash, n_hash));
    } else if constexpr (std::is_same_v<DType, double>) {
        CUBLAS_CHECK(cublasDgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, n_hash, n_rows, n_cols,
                                 &alpha, P, n_cols, X, n_cols, &beta, X_hash, n_hash));
    } else {
        throw std::invalid_argument("Expected float or double dtype");
    }
}

} // namespace detail
} // namespace rplsh
} // namespace culsh
