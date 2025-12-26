#include "../core/candidates.cuh"
#include "../core/fit.cuh"
#include "../core/fit_query.cuh"
#include "../core/index.cuh"
#include "../core/query.cuh"
#include "hash.cuh"
#include "projections.cuh"
#include <cuda_runtime.h>
#include <culsh/rplsh/params.hpp>
#include <culsh/rplsh/rplsh.hpp>
#include <curand.h>

namespace culsh {
namespace rplsh {

using Index = core::Index;
using Candidates = core::Candidates;

Index fit(cublasHandle_t cublas_handle, cudaStream_t stream, const float* X, int n_samples,
          int n_features, const RPLSHParams& params) {

    // Allocate projection matrix
    size_t P_size = static_cast<size_t>(params.n_hash_tables) * params.n_hashes * n_features;
    float* P;
    CUDA_CHECK(cudaMalloc(&P, P_size * sizeof(float)));

    // Allocate X_hash
    float* X_hash;
    CUDA_CHECK(cudaMalloc(&X_hash, static_cast<size_t>(n_samples) * params.n_hash_tables *
                                       params.n_hashes * sizeof(float)));

    // Generate random projections and hash X
    detail::generate_random_projections<float>(stream, params.n_hash_tables * params.n_hashes,
                                               n_features, params.seed, P);
    detail::hash<float>(cublas_handle, stream, X, P, n_samples, n_features, params.n_hash_tables,
                        params.n_hashes, X_hash);

    // Compute binary signatures from X_hash
    int8_t* X_sig;
    CUDA_CHECK(cudaMalloc(&X_sig, static_cast<size_t>(n_samples) * params.n_hash_tables *
                                      params.n_hashes * sizeof(int8_t)));
    detail::compute_signatures<float>(stream, X_hash, n_samples, params.n_hash_tables,
                                      params.n_hashes, X_sig);
    CUDA_CHECK(cudaFree(X_hash));

    // Build index
    auto index =
        core::detail::fit_index(stream, X_sig, n_samples, params.n_hash_tables, params.n_hashes);
    CUDA_CHECK(cudaFree(X_sig));

    // Store projection matrix and metadata in index
    index.P = P;
    index.n_features = n_features;
    index.seed = params.seed;
    index.is_double = false;

    return index;
}

Index fit(cublasHandle_t cublas_handle, cudaStream_t stream, const double* X, int n_samples,
          int n_features, const RPLSHParams& params) {

    // Allocate projection matrix
    size_t P_size = static_cast<size_t>(params.n_hash_tables) * params.n_hashes * n_features;
    double* P;
    CUDA_CHECK(cudaMalloc(&P, P_size * sizeof(double)));

    // Allocate X_hash
    double* X_hash;
    CUDA_CHECK(cudaMalloc(&X_hash, static_cast<size_t>(n_samples) * params.n_hash_tables *
                                       params.n_hashes * sizeof(double)));

    // Generate random projections and hash X
    detail::generate_random_projections<double>(stream, params.n_hash_tables * params.n_hashes,
                                                n_features, params.seed, P);
    detail::hash<double>(cublas_handle, stream, X, P, n_samples, n_features, params.n_hash_tables,
                         params.n_hashes, X_hash);

    // Compute binary signatures from X_hash
    int8_t* X_sig;
    CUDA_CHECK(cudaMalloc(&X_sig, static_cast<size_t>(n_samples) * params.n_hash_tables *
                                      params.n_hashes * sizeof(int8_t)));
    detail::compute_signatures<double>(stream, X_hash, n_samples, params.n_hash_tables,
                                       params.n_hashes, X_sig);
    CUDA_CHECK(cudaFree(X_hash));

    // Build index
    auto index =
        core::detail::fit_index(stream, X_sig, n_samples, params.n_hash_tables, params.n_hashes);
    CUDA_CHECK(cudaFree(X_sig));

    // Store projection matrix and metadata in index
    index.P = P;
    index.n_features = n_features;
    index.seed = params.seed;
    index.is_double = true;

    return index;
}

Candidates query(cublasHandle_t cublas_handle, cudaStream_t stream, const float* Q, int n_queries,
                 const Index& index) {

    const float* P = index.P_float();
    int n_features = index.n_features;
    int n_hash_tables = index.n_hash_tables;
    int n_hashes = index.n_hashes;

    // Allocate Q_hash and hash Q
    float* Q_hash;
    CUDA_CHECK(cudaMalloc(&Q_hash, static_cast<size_t>(n_queries) * n_hash_tables * n_hashes *
                                       sizeof(float)));

    detail::hash<float>(cublas_handle, stream, Q, P, n_queries, n_features, n_hash_tables, n_hashes,
                        Q_hash);

    // Compute binary signatures from Q_hash
    int8_t* Q_sig;
    CUDA_CHECK(cudaMalloc(&Q_sig, static_cast<size_t>(n_queries) * n_hash_tables * n_hashes *
                                      sizeof(int8_t)));
    detail::compute_signatures<float>(stream, Q_hash, n_queries, n_hash_tables, n_hashes, Q_sig);
    CUDA_CHECK(cudaFree(Q_hash));

    // Query index for candidate indices
    auto candidates =
        core::detail::query_index(stream, Q_sig, n_queries, n_hash_tables, n_hashes, &index);
    CUDA_CHECK(cudaFree(Q_sig));

    return candidates;
}

Candidates query(cublasHandle_t cublas_handle, cudaStream_t stream, const double* Q, int n_queries,
                 const Index& index) {

    const double* P = index.P_double();
    int n_features = index.n_features;
    int n_hash_tables = index.n_hash_tables;
    int n_hashes = index.n_hashes;

    // Allocate Q_hash and hash Q
    double* Q_hash;
    CUDA_CHECK(cudaMalloc(&Q_hash, static_cast<size_t>(n_queries) * n_hash_tables * n_hashes *
                                       sizeof(double)));

    detail::hash<double>(cublas_handle, stream, Q, P, n_queries, n_features, n_hash_tables,
                         n_hashes, Q_hash);

    // Compute binary signatures from Q_hash
    int8_t* Q_sig;
    CUDA_CHECK(cudaMalloc(&Q_sig, static_cast<size_t>(n_queries) * n_hash_tables * n_hashes *
                                      sizeof(int8_t)));
    detail::compute_signatures<double>(stream, Q_hash, n_queries, n_hash_tables, n_hashes, Q_sig);
    CUDA_CHECK(cudaFree(Q_hash));

    // Query index for candidate indices
    auto candidates =
        core::detail::query_index(stream, Q_sig, n_queries, n_hash_tables, n_hashes, &index);
    CUDA_CHECK(cudaFree(Q_sig));

    return candidates;
}

Candidates fit_query(cublasHandle_t cublas_handle, cudaStream_t stream, const float* X,
                     int n_samples, int n_features, const RPLSHParams& params) {

    // Allocate projection matrix
    size_t P_size = static_cast<size_t>(params.n_hash_tables) * params.n_hashes * n_features;
    float* P;
    CUDA_CHECK(cudaMalloc(&P, P_size * sizeof(float)));

    // Allocate X_hash
    float* X_hash;
    CUDA_CHECK(cudaMalloc(&X_hash, static_cast<size_t>(n_samples) * params.n_hash_tables *
                                       params.n_hashes * sizeof(float)));

    // Generate random projections and hash X
    detail::generate_random_projections<float>(stream, params.n_hash_tables * params.n_hashes,
                                               n_features, params.seed, P);
    detail::hash<float>(cublas_handle, stream, X, P, n_samples, n_features, params.n_hash_tables,
                        params.n_hashes, X_hash);

    // Compute binary signatures from X_hash
    int8_t* X_sig;
    CUDA_CHECK(cudaMalloc(&X_sig, static_cast<size_t>(n_samples) * params.n_hash_tables *
                                      params.n_hashes * sizeof(int8_t)));
    detail::compute_signatures<float>(stream, X_hash, n_samples, params.n_hash_tables,
                                      params.n_hashes, X_sig);
    CUDA_CHECK(cudaFree(X_hash));

    // Fit index and query candidates directly (X_sig == Q_sig)
    auto candidates =
        core::detail::fit_query(stream, X_sig, n_samples, params.n_hash_tables, params.n_hashes);
    CUDA_CHECK(cudaFree(X_sig));

    return candidates;
}

Candidates fit_query(cublasHandle_t cublas_handle, cudaStream_t stream, const double* X,
                     int n_samples, int n_features, const RPLSHParams& params) {

    // Allocate projection matrix
    size_t P_size = static_cast<size_t>(params.n_hash_tables) * params.n_hashes * n_features;
    double* P;
    CUDA_CHECK(cudaMalloc(&P, P_size * sizeof(double)));

    // Allocate X_hash
    double* X_hash;
    CUDA_CHECK(cudaMalloc(&X_hash, static_cast<size_t>(n_samples) * params.n_hash_tables *
                                       params.n_hashes * sizeof(double)));

    // Generate random projections and hash X
    detail::generate_random_projections<double>(stream, params.n_hash_tables * params.n_hashes,
                                                n_features, params.seed, P);
    detail::hash<double>(cublas_handle, stream, X, P, n_samples, n_features, params.n_hash_tables,
                         params.n_hashes, X_hash);

    // Compute binary signatures from X_hash
    int8_t* X_sig;
    CUDA_CHECK(cudaMalloc(&X_sig, static_cast<size_t>(n_samples) * params.n_hash_tables *
                                      params.n_hashes * sizeof(int8_t)));
    detail::compute_signatures<double>(stream, X_hash, n_samples, params.n_hash_tables,
                                       params.n_hashes, X_sig);
    CUDA_CHECK(cudaFree(X_hash));

    // Fit index and query candidates directly (X_sig == Q_sig)
    auto candidates =
        core::detail::fit_query(stream, X_sig, n_samples, params.n_hash_tables, params.n_hashes);
    CUDA_CHECK(cudaFree(X_sig));

    return candidates;
}

Candidates query_batched(cublasHandle_t cublas_handle, cudaStream_t stream, const float* Q,
                         int n_queries, const Index& index, int batch_size) {
    if (batch_size <= 0 || batch_size >= n_queries) {
        return query(cublas_handle, stream, Q, n_queries, index);
    }

    Candidates result;
    for (int start = 0; start < n_queries; start += batch_size) {
        int end = std::min(start + batch_size, n_queries);
        int batch_n = end - start;

        // Query this batch and merge into result
        const float* Q_batch = Q + static_cast<size_t>(start) * index.n_features;
        Candidates batch_candidates = query(cublas_handle, stream, Q_batch, batch_n, index);
        result.merge(stream, std::move(batch_candidates));
    }

    return result;
}

Candidates query_batched(cublasHandle_t cublas_handle, cudaStream_t stream, const double* Q,
                         int n_queries, const Index& index, int batch_size) {
    if (batch_size <= 0 || batch_size >= n_queries) {
        return query(cublas_handle, stream, Q, n_queries, index);
    }

    Candidates result;
    for (int start = 0; start < n_queries; start += batch_size) {
        int end = std::min(start + batch_size, n_queries);
        int batch_n = end - start;

        // Query this batch and merge into result
        const double* Q_batch = Q + static_cast<size_t>(start) * index.n_features;
        Candidates batch_candidates = query(cublas_handle, stream, Q_batch, batch_n, index);
        result.merge(stream, std::move(batch_candidates));
    }

    return result;
}

} // namespace rplsh
} // namespace culsh
