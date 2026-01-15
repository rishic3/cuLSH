#include "../core/candidates.cuh"
#include "../core/fit.cuh"
#include "../core/fit_query.cuh"
#include "../core/index.cuh"
#include "../core/query.cuh"
#include "hash.cuh"
#include "index.cuh"
#include <cuda_runtime.h>
#include <culsh/pslsh/params.hpp>
#include <culsh/pslsh/pslsh.hpp>
#include <curand.h>

namespace culsh {
namespace pslsh {

using Candidates = core::Candidates;

Index fit(cublasHandle_t cublas_handle, cudaStream_t stream, const float* X, int n_samples,
          int n_features, const PSLSHParams& params) {
    // Allocate projections/biases
    size_t P_size = static_cast<size_t>(params.n_hash_tables) * params.n_hashes * n_features;
    float* P = nullptr;
    float* b = nullptr;
    CUDA_CHECK_ALLOC(cudaMalloc(&P, P_size * sizeof(float)));
    CUDA_CHECK_ALLOC(cudaMalloc(&b, params.n_hash_tables * params.n_hashes * sizeof(float)));

    // Generate random projections/biases
    detail::generate_random_projections_biases<float>(
        stream, params.n_hash_tables * params.n_hashes, n_features, params.window_size, params.seed,
        P, b);

    int32_t* X_sig = nullptr;
    CUDA_CHECK_ALLOC(cudaMalloc(&X_sig, static_cast<size_t>(n_samples) * params.n_hash_tables *
                                      params.n_hashes * sizeof(int32_t)));
    detail::hash<float>(cublas_handle, stream, X, P, b, n_samples, n_features, params.n_hash_tables,
                        params.n_hashes, X_sig);

    // Build index
    int sig_nbytes = params.n_hashes * static_cast<int>(sizeof(int32_t));
    core::Index core_index = core::detail::fit_index(stream, X_sig, n_samples, params.n_hash_tables,
                                                     params.n_hashes, sig_nbytes);
    CUDA_CHECK(cudaFree(X_sig));

    // Wrap core index and store projections/biases + metadata
    Index index;
    index.core = std::move(core_index);
    index.core.n_features = n_features;
    index.core.seed = params.seed;
    index.P = P;
    index.b = b;
    index.is_double = false;
    return index;
}

Index fit(cublasHandle_t cublas_handle, cudaStream_t stream, const double* X, int n_samples,
          int n_features, const PSLSHParams& params) {
    // Allocate projections/biases
    size_t P_size = static_cast<size_t>(params.n_hash_tables) * params.n_hashes * n_features;
    double* P = nullptr;
    double* b = nullptr;
    CUDA_CHECK_ALLOC(cudaMalloc(&P, P_size * sizeof(double)));
    CUDA_CHECK_ALLOC(cudaMalloc(&b, params.n_hash_tables * params.n_hashes * sizeof(double)));

    // Generate random projections/biases
    detail::generate_random_projections_biases<double>(
        stream, params.n_hash_tables * params.n_hashes, n_features, params.window_size, params.seed,
        P, b);

    int32_t* X_sig = nullptr;
    CUDA_CHECK_ALLOC(cudaMalloc(&X_sig, static_cast<size_t>(n_samples) * params.n_hash_tables *
                                      params.n_hashes * sizeof(int32_t)));
    detail::hash<double>(cublas_handle, stream, X, P, b, n_samples, n_features,
                         params.n_hash_tables, params.n_hashes, X_sig);

    // Build index
    int sig_nbytes = params.n_hashes * static_cast<int>(sizeof(int32_t));
    core::Index core_index = core::detail::fit_index(stream, X_sig, n_samples, params.n_hash_tables,
                                                     params.n_hashes, sig_nbytes);
    CUDA_CHECK(cudaFree(X_sig));

    // Wrap core index and store projections/biases + metadata
    Index index;
    index.core = std::move(core_index);
    index.core.n_features = n_features;
    index.core.seed = params.seed;
    index.P = P;
    index.b = b;
    index.is_double = true;
    return index;
}

Candidates query(cublasHandle_t cublas_handle, cudaStream_t stream, const float* Q, int n_queries,
                 const Index& index) {
    const float* P = index.P_float();
    const float* b = index.b_float();
    int n_features = index.core.n_features;
    int n_hash_tables = index.core.n_hash_tables;
    int n_hashes = index.core.n_hashes;

    // Hash Q to signatures
    int32_t* Q_sig = nullptr;
    CUDA_CHECK_ALLOC(cudaMalloc(&Q_sig, static_cast<size_t>(n_queries) * n_hash_tables * n_hashes *
                                      sizeof(int32_t)));
    detail::hash<float>(cublas_handle, stream, Q, P, b, n_queries, n_features, n_hash_tables,
                        n_hashes, Q_sig);

    // Query index for candidate indices
    int sig_nbytes = n_hashes * static_cast<int>(sizeof(int32_t));
    auto candidates = core::detail::query_index(stream, Q_sig, n_queries, sig_nbytes, &index.core);
    CUDA_CHECK(cudaFree(Q_sig));

    return candidates;
}

Candidates query(cublasHandle_t cublas_handle, cudaStream_t stream, const double* Q, int n_queries,
                 const Index& index) {
    const double* P = index.P_double();
    const double* b = index.b_double();
    int n_features = index.core.n_features;
    int n_hash_tables = index.core.n_hash_tables;
    int n_hashes = index.core.n_hashes;

    // Hash Q to signatures
    int32_t* Q_sig = nullptr;
    CUDA_CHECK_ALLOC(cudaMalloc(&Q_sig, static_cast<size_t>(n_queries) * n_hash_tables * n_hashes *
                                      sizeof(int32_t)));
    detail::hash<double>(cublas_handle, stream, Q, P, b, n_queries, n_features, n_hash_tables,
                         n_hashes, Q_sig);

    // Query index for candidate indices
    int sig_nbytes = n_hashes * static_cast<int>(sizeof(int32_t));
    auto candidates = core::detail::query_index(stream, Q_sig, n_queries, sig_nbytes, &index.core);
    CUDA_CHECK(cudaFree(Q_sig));

    return candidates;
}

Candidates fit_query(cublasHandle_t cublas_handle, cudaStream_t stream, const float* X,
                     int n_samples, int n_features, const PSLSHParams& params) {
    // Allocate projections/biases
    size_t P_size = static_cast<size_t>(params.n_hash_tables) * params.n_hashes * n_features;
    float* P = nullptr;
    float* b = nullptr;
    CUDA_CHECK_ALLOC(cudaMalloc(&P, P_size * sizeof(float)));
    CUDA_CHECK_ALLOC(cudaMalloc(&b, params.n_hash_tables * params.n_hashes * sizeof(float)));

    // Generate random projections/biases
    detail::generate_random_projections_biases<float>(
        stream, params.n_hash_tables * params.n_hashes, n_features, params.window_size, params.seed,
        P, b);

    // Hash X to signatures
    int32_t* X_sig = nullptr;
    CUDA_CHECK_ALLOC(cudaMalloc(&X_sig, static_cast<size_t>(n_samples) * params.n_hash_tables *
                                      params.n_hashes * sizeof(int32_t)));
    detail::hash<float>(cublas_handle, stream, X, P, b, n_samples, n_features, params.n_hash_tables,
                        params.n_hashes, X_sig);

    // Fit index and query candidates directly (X_sig == Q_sig)
    auto candidates =
        core::detail::fit_query(stream, X_sig, n_samples, params.n_hash_tables, params.n_hashes,
                                params.n_hashes * static_cast<int>(sizeof(int32_t)));
    
    CUDA_CHECK(cudaFree(X_sig));
    CUDA_CHECK(cudaFree(P));
    CUDA_CHECK(cudaFree(b));

    return candidates;
}

Candidates fit_query(cublasHandle_t cublas_handle, cudaStream_t stream, const double* X,
                     int n_samples, int n_features, const PSLSHParams& params) {
    // Allocate projections/biases
    size_t P_size = static_cast<size_t>(params.n_hash_tables) * params.n_hashes * n_features;
    double* P = nullptr;
    double* b = nullptr;
    CUDA_CHECK_ALLOC(cudaMalloc(&P, P_size * sizeof(double)));
    CUDA_CHECK_ALLOC(cudaMalloc(&b, params.n_hash_tables * params.n_hashes * sizeof(double)));

    // Generate random projections/biases
    detail::generate_random_projections_biases<double>(
        stream, params.n_hash_tables * params.n_hashes, n_features, params.window_size, params.seed,
        P, b);

    // Hash X to signatures
    int32_t* X_sig = nullptr;
    CUDA_CHECK_ALLOC(cudaMalloc(&X_sig, static_cast<size_t>(n_samples) * params.n_hash_tables *
                                      params.n_hashes * sizeof(int32_t)));
    detail::hash<double>(cublas_handle, stream, X, P, b, n_samples, n_features,
                         params.n_hash_tables, params.n_hashes, X_sig);

    // Fit index and query candidates directly (X_sig == Q_sig)
    auto candidates =
        core::detail::fit_query(stream, X_sig, n_samples, params.n_hash_tables, params.n_hashes,
                                params.n_hashes * static_cast<int>(sizeof(int32_t)));

    CUDA_CHECK(cudaFree(X_sig));
    CUDA_CHECK(cudaFree(P));
    CUDA_CHECK(cudaFree(b));

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
        const float* Q_batch = Q + static_cast<size_t>(start) * index.core.n_features;
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
        const double* Q_batch = Q + static_cast<size_t>(start) * index.core.n_features;
        Candidates batch_candidates = query(cublas_handle, stream, Q_batch, batch_n, index);
        result.merge(stream, std::move(batch_candidates));
    }

    return result;
}

} // namespace pslsh
} // namespace culsh
