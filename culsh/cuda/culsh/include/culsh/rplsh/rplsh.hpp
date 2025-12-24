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
 * @brief Fit the RPLSH index
 *
 * @param[in] cublas_handle cuBLAS handle
 * @param[in] stream CUDA stream
 * @param[in] X Input data matrix (row-major)
 * @param[in] n_samples Number of data points
 * @param[in] n_features Number of features
 * @param[in] params LSH params
 * @return Index containing hash tables and projection matrix
 */
Index fit(cublasHandle_t cublas_handle, cudaStream_t stream, const float* X, int n_samples,
          int n_features, const RPLSHParams& params);

/**
 * @brief Fit the RPLSH index (double precision)
 *
 * @param[in] cublas_handle cuBLAS handle
 * @param[in] stream CUDA stream
 * @param[in] X Input data matrix (row-major)
 * @param[in] n_samples Number of data points
 * @param[in] n_features Number of features
 * @param[in] params LSH params
 * @return Index containing hash tables and projection matrix
 */
Index fit(cublasHandle_t cublas_handle, cudaStream_t stream, const double* X, int n_samples,
          int n_features, const RPLSHParams& params);

/**
 * @brief Query the RPLSH index for candidate neighbor indices
 *
 * @param[in] cublas_handle cuBLAS handle
 * @param[in] stream CUDA stream
 * @param[in] Q Query vectors (row-major)
 * @param[in] n_queries Number of query points
 * @param[in] index RPLSH index (contains projection matrix)
 * @return Candidates containing candidate indices for each query
 */
Candidates query(cublasHandle_t cublas_handle, cudaStream_t stream, const float* Q, int n_queries,
                 const Index& index);

/**
 * @brief Query the RPLSH index for candidate neighbor indices (double precision)
 *
 * @param[in] cublas_handle cuBLAS handle
 * @param[in] stream CUDA stream
 * @param[in] Q Query vectors (row-major)
 * @param[in] n_queries Number of query points
 * @param[in] index RPLSH index (contains projection matrix)
 * @return Candidates containing candidate indices for each query
 */
Candidates query(cublasHandle_t cublas_handle, cudaStream_t stream, const double* Q, int n_queries,
                 const Index& index);

/**
 * @brief Simultaneously fit and query the RPLSH index
 *
 * @param[in] cublas_handle cuBLAS handle
 * @param[in] stream CUDA stream
 * @param[in] X Input data matrix (row-major)
 * @param[in] n_samples Number of data points
 * @param[in] n_features Number of features
 * @param[in] params LSH params
 * @return Candidates containing candidate indices for each query
 */
Candidates fit_query(cublasHandle_t cublas_handle, cudaStream_t stream, const float* X,
                     int n_samples, int n_features, const RPLSHParams& params);

/**
 * @brief Simultaneously fit and query the RPLSH index (double precision)
 *
 * @param[in] cublas_handle cuBLAS handle
 * @param[in] stream CUDA stream
 * @param[in] X Input data matrix (row-major)
 * @param[in] n_samples Number of data points
 * @param[in] n_features Number of features
 * @param[in] params LSH params
 * @return Candidates containing candidate indices for each query
 */
Candidates fit_query(cublasHandle_t cublas_handle, cudaStream_t stream, const double* X,
                     int n_samples, int n_features, const RPLSHParams& params);

/**
 * @brief Query the RPLSH index in batches
 *
 * @param[in] cublas_handle cuBLAS handle
 * @param[in] stream CUDA stream
 * @param[in] Q Query vectors (row-major)
 * @param[in] n_queries Number of query points
 * @param[in] index RPLSH index (contains projection matrix)
 * @param[in] batch_size Number of queries per batch
 * @return Candidates containing candidate indices for each query
 */
Candidates query_batched(cublasHandle_t cublas_handle, cudaStream_t stream, const float* Q,
                         int n_queries, const Index& index, int batch_size);

/**
 * @brief Query the RPLSH index in batches (double precision)
 *
 * @param[in] cublas_handle cuBLAS handle
 * @param[in] stream CUDA stream
 * @param[in] Q Query vectors (row-major)
 * @param[in] n_queries Number of query points
 * @param[in] index RPLSH index (contains projection matrix)
 * @param[in] batch_size Number of queries per batch
 * @return Candidates containing candidate indices for each query
 */
Candidates query_batched(cublasHandle_t cublas_handle, cudaStream_t stream, const double* Q,
                         int n_queries, const Index& index, int batch_size);

} // namespace rplsh
} // namespace culsh
