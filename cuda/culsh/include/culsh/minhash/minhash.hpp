#pragma once

#include "params.hpp"
#include <cublas_v2.h>

namespace culsh {

// Forward declarations
namespace core {
struct Candidates;
} // namespace core

namespace minhash {

struct Index;
using Candidates = core::Candidates;

/**
 * @brief Fit the MinHash LSH index
 *
 * @param[in] stream CUDA stream
 * @param[in] X_indices Indices for input CSR matrix (nnz)
 * @param[in] X_indptr Indptr for input CSR matrix (n_samples + 1)
 * @param[in] n_samples Number of data points
 * @param[in] n_features Number of features
 * @param[in] params LSH params
 * @return Index containing hash tables and projection matrix
 */
Index fit(cudaStream_t stream, const int* X_indices, const int* X_indptr, int n_samples,
          int n_features, const MinHashParams& params);

/**
 * @brief Query the MinHash LSH index for candidate neighbor indices
 *
 * @param[in] stream CUDA stream
 * @param[in] Q_indices Indices for query CSR matrix (nnz)
 * @param[in] Q_indptr Indptr for query CSR matrix (n_queries + 1)
 * @param[in] n_queries Number of query points
 * @param[in] index MinHash LSH index (contains projection matrix)
 * @return Candidates containing candidate indices for each query
 */
Candidates query(cudaStream_t stream, const int* Q_indices, const int* Q_indptr, int n_queries,
                 const Index& index);

/**
 * @brief Simultaneously fit and query the MinHash LSH index
 *
 * @param[in] stream CUDA stream
 * @param[in] X_indices Indices for input CSR matrix (nnz)
 * @param[in] X_indptr Indptr for input CSR matrix (n_samples + 1)
 * @param[in] n_samples Number of data points
 * @param[in] n_features Number of features
 * @param[in] params LSH params
 * @return Candidates containing candidate indices for each query
 */
Candidates fit_query(cudaStream_t stream, const int* X_indices, const int* X_indptr, int n_samples,
                     int n_features, const MinHashParams& params);

} // namespace minhash
} // namespace culsh
