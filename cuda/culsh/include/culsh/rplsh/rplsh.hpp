#pragma once

#include "params.hpp"
#include <cublas_v2.h>

namespace culsh {
namespace rplsh {

/**
 * @brief Random Projection LSH index
 */
struct Index;

/**
 * @brief Random Projection LSH candidates
 */
struct Candidates;

/**
 * @brief Fit the Random Projection LSH model
 *
 * @param[in] cublas_handle cuBLAS handle
 * @param[in] stream CUDA stream
 * @param[in] X Input data matrix (row-major)
 * @param[in] n_samples Number of data points
 * @param[in] n_features Number of features
 * @param[in] params LSH params
 * @param[out] P Random projection matrix
 * @param[out] index RPLSH index
 */
Index fit(cublasHandle_t cublas_handle, cudaStream_t stream, const float* X, int n_samples,
          int n_features, const RPLSHParams& params, float* P);

/**
 * @brief Fit the Random Projection LSH model
 *
 * @param[in] cublas_handle cuBLAS handle
 * @param[in] stream CUDA stream
 * @param[in] X Input data matrix (row-major)
 * @param[in] n_samples Number of data points
 * @param[in] n_features Number of features
 * @param[in] params LSH params
 * @param[out] P Random projection matrix
 * @param[out] index RPLSH index
 */
Index fit(cublasHandle_t cublas_handle, cudaStream_t stream, const double* X, int n_samples,
          int n_features, const RPLSHParams& params, double* P);

/**
 * @brief Query the Random Projection LSH model for candidate neighbor indices
 *
 * @param[in] cublas_handle cuBLAS handle
 * @param[in] stream CUDA stream
 * @param[in] Q Query vectors (row-major)
 * @param[in] n_queries Number of query points
 * @param[in] n_features Number of features
 * @param[in] params LSH params
 * @param[in] P Random projection matrix
 * @param[in] index RPLSH hash table index
 * @param[out] candidates Output candidate indices
 */
Candidates query_indices(cublasHandle_t cublas_handle, cudaStream_t stream, const float* Q,
                         int n_queries, int n_features, const RPLSHParams& params, const float* P,
                         const Index* index);

/**
 * @brief Query the Random Projection LSH model for candidate neighbor indices
 *
 * @param[in] cublas_handle cuBLAS handle
 * @param[in] stream CUDA stream
 * @param[in] Q Query vectors (row-major)
 * @param[in] n_queries Number of query points
 * @param[in] n_features Number of features
 * @param[in] params LSH params
 * @param[in] P Random projection matrix
 * @param[in] index RPLSH hash table index
 * @param[out] candidates Output candidate indices
 */
Candidates query_indices(cublasHandle_t cublas_handle, cudaStream_t stream, const double* Q,
                         int n_queries, int n_features, const RPLSHParams& params, const double* P,
                         const Index* index);

/**
 * @brief Query the Random Projection LSH model for candidate neighbor vectors
 *
 * @param[in] cublas_handle cuBLAS handle
 * @param[in] stream CUDA stream
 * @param[in] Q Query vectors (row-major)
 * @param[in] n_queries Number of query points
 * @param[in] n_features Number of features
 * @param[in] params LSH params
 * @param[in] P Random projection matrix
 * @param[in] index RPLSH hash table index
 * @param[in] X_stored Fitted data vectors
 * @param[out] candidate_vectors Output candidate vectors
 */
Candidates query_vectors(cublasHandle_t cublas_handle, cudaStream_t stream, const float* Q,
                         int n_queries, int n_features, const RPLSHParams& params, const float* P,
                         const Index* index, const float* X_stored);

/**
 * @brief Query the Random Projection LSH model for candidate neighbor vectors
 *
 * @param[in] cublas_handle cuBLAS handle
 * @param[in] stream CUDA stream
 * @param[in] Q Query vectors (row-major)
 * @param[in] n_queries Number of query points
 * @param[in] n_features Number of features
 * @param[in] params LSH params
 * @param[in] P Random projection matrix
 * @param[in] index RPLSH hash table index
 * @param[in] X_stored Fitted data vectors
 * @param[out] candidate_vectors Output candidate vectors
 */
Candidates query_vectors(cublasHandle_t cublas_handle, cudaStream_t stream, const double* Q,
                         int n_queries, int n_features, const RPLSHParams& params, const double* P,
                         const Index* index, const double* X_stored);

} // namespace rplsh
} // namespace culsh
