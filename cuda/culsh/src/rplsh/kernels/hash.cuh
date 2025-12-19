#pragma once

#include "../utils/utils.cuh"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdexcept>

namespace culsh {
namespace rplsh {
namespace detail {

/**
 * @brief Hash the input vectors X using random projection matrix P
 * @param[in] cublas_handle cuBLAS handle
 * @param[in] stream CUDA stream
 * @param[in] X Input vectors (n_samples x n_features)
 * @param[in] P Random projection matrix (n_hash_tables * n_projections x n_features)
 * @param[in] n_samples Number of input vectors
 * @param[in] n_features Dimensionality of each input vector
 * @param[in] n_hash_tables Number of hash tables
 * @param[in] n_projections Number of projections
 * @param[out] X_hash Hashed input vectors (n_samples x n_hash_tables * n_projections)
 */
template <typename DType>
void hash(cublasHandle_t cublas_handle, cudaStream_t stream, const DType* X, const DType* P,
          int n_samples, int n_features, int n_hash_tables, int n_projections, DType* X_hash) {

    cublasSetStream(cublas_handle, stream);

    const DType alpha = DType(1.0);
    const DType beta = DType(0.0);
    const int n_total_buckets = n_hash_tables * n_projections;

    // given row-major X, P, compute X * P^T = X_hash.
    // in col-major (used by cuBLAS - denoting with _c), we have X_c = X^T, P_c = P^T.
    // thus to get X_hash in row-major, compute P_c^T * X_c = X_hash_c^T = X_hash.
    if constexpr (std::is_same_v<DType, float>) {
        CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, n_total_buckets,
                                 n_samples, n_features, &alpha, P, n_features, X, n_features, &beta,
                                 X_hash, n_total_buckets));
    } else if constexpr (std::is_same_v<DType, double>) {
        CUBLAS_CHECK(cublasDgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, n_total_buckets,
                                 n_samples, n_features, &alpha, P, n_features, X, n_features, &beta,
                                 X_hash, n_total_buckets));
    } else {
        throw std::invalid_argument("Expected float or double dtype");
    }
}

} // namespace detail
} // namespace rplsh
} // namespace culsh
