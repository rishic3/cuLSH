#include "candidates.cuh"
#include "index.cuh"
#include "kernels/build_index.cuh"
#include "kernels/hash.cuh"
#include "kernels/projections.cuh"
#include "kernels/query_index.cuh"
#include <cuda_runtime.h>
#include <culsh/rplsh/params.hpp>
#include <culsh/rplsh/rplsh.hpp>
#include <curand.h>

namespace culsh {
namespace rplsh {

Index fit(cublasHandle_t cublas_handle, cudaStream_t stream, const float* X, int n_samples,
          int n_features, const RPLSHParams& params, float* P) {

    // allocate X_hash
    float* X_hash;
    CUDA_CHECK(cudaMalloc(&X_hash, static_cast<size_t>(n_samples) * params.n_hash_tables *
                                       params.n_projections * sizeof(float)));

    // generate random projections and hash X
    detail::generate_random_projections<float>(stream, params.n_hash_tables * params.n_projections,
                                               n_features, params.seed, P);
    detail::hash<float>(cublas_handle, stream, X, P, n_samples, n_features, params.n_hash_tables,
                        params.n_projections, X_hash);

    // compute binary signatures from X_hash
    int8_t* X_sig;
    CUDA_CHECK(cudaMalloc(&X_sig, static_cast<size_t>(n_samples) * params.n_hash_tables *
                                      params.n_projections * sizeof(int8_t)));
    detail::compute_signatures<float>(stream, X_hash, n_samples, params.n_hash_tables,
                                      params.n_projections, X_sig);
    CUDA_CHECK(cudaFree(X_hash)); // done with X_hash

    // build and return index
    auto index =
        detail::build_index(stream, X_sig, n_samples, params.n_hash_tables, params.n_projections);
    CUDA_CHECK(cudaFree(X_sig));

    return index;
}

Index fit(cublasHandle_t cublas_handle, cudaStream_t stream, const double* X, int n_samples,
          int n_features, const RPLSHParams& params, double* P) {

    // allocate X_hash
    double* X_hash;
    CUDA_CHECK(cudaMalloc(&X_hash, static_cast<size_t>(n_samples) * params.n_hash_tables *
                                       params.n_projections * sizeof(double)));

    // generate random projections and hash X
    detail::generate_random_projections<double>(stream, params.n_hash_tables * params.n_projections,
                                                n_features, params.seed, P);
    detail::hash<double>(cublas_handle, stream, X, P, n_samples, n_features, params.n_hash_tables,
                         params.n_projections, X_hash);

    // compute binary signatures from X_hash
    int8_t* X_sig;
    CUDA_CHECK(cudaMalloc(&X_sig, static_cast<size_t>(n_samples) * params.n_hash_tables *
                                      params.n_projections * sizeof(int8_t)));
    detail::compute_signatures<double>(stream, X_hash, n_samples, params.n_hash_tables,
                                       params.n_projections, X_sig);
    CUDA_CHECK(cudaFree(X_hash)); // done with X_hash

    // build and return index
    auto index =
        detail::build_index(stream, X_sig, n_samples, params.n_hash_tables, params.n_projections);
    CUDA_CHECK(cudaFree(X_sig));

    return index;
}

Candidates query_indices(cublasHandle_t cublas_handle, cudaStream_t stream, const float* Q,
                         int n_queries, int n_features, const RPLSHParams& params, const float* P,
                         const Index* index) {

    // allocate Q_hash and hash Q
    float* Q_hash;
    CUDA_CHECK(cudaMalloc(&Q_hash, static_cast<size_t>(n_queries) * params.n_hash_tables *
                                       params.n_projections * sizeof(float)));

    detail::hash<float>(cublas_handle, stream, Q, P, n_queries, n_features, params.n_hash_tables,
                        params.n_projections, Q_hash);

    // compute binary signatures from Q_hash
    int8_t* Q_sig;
    CUDA_CHECK(cudaMalloc(&Q_sig, static_cast<size_t>(n_queries) * params.n_hash_tables *
                                      params.n_projections * sizeof(int8_t)));
    detail::compute_signatures<float>(stream, Q_hash, n_queries, params.n_hash_tables,
                                      params.n_projections, Q_sig);
    CUDA_CHECK(cudaFree(Q_hash));

    // query index for candidate indices
    auto candidates = detail::query_index(stream, Q_sig, n_queries, params.n_hash_tables,
                                          params.n_projections, index);
    CUDA_CHECK(cudaFree(Q_sig));

    return candidates;
}

Candidates query_indices(cublasHandle_t cublas_handle, cudaStream_t stream, const double* Q,
                         int n_queries, int n_features, const RPLSHParams& params, const double* P,
                         const Index* index) {

    // allocate Q_hash and hash Q
    double* Q_hash;
    CUDA_CHECK(cudaMalloc(&Q_hash, static_cast<size_t>(n_queries) * params.n_hash_tables *
                                       params.n_projections * sizeof(double)));

    detail::hash<double>(cublas_handle, stream, Q, P, n_queries, n_features, params.n_hash_tables,
                         params.n_projections, Q_hash);

    // compute binary signatures from Q_hash
    int8_t* Q_sig;
    CUDA_CHECK(cudaMalloc(&Q_sig, static_cast<size_t>(n_queries) * params.n_hash_tables *
                                      params.n_projections * sizeof(int8_t)));
    detail::compute_signatures<double>(stream, Q_hash, n_queries, params.n_hash_tables,
                                       params.n_projections, Q_sig);
    CUDA_CHECK(cudaFree(Q_hash));

    // query index for candidate indices
    auto candidates = detail::query_index(stream, Q_sig, n_queries, params.n_hash_tables,
                                          params.n_projections, index);
    CUDA_CHECK(cudaFree(Q_sig));

    return candidates;
}

Candidates query_vectors(cublasHandle_t cublas_handle, cudaStream_t stream, const float* Q,
                         int n_queries, int n_features, const RPLSHParams& params, const float* P,
                         const Index* index, const float* X_stored) {

    // TODO: Hash query vectors using projections
    // TODO: Query index for candidate indices
    // TODO: Retrieve actual vectors from X_stored using indices
    // TODO: Return results in candidate_vectors and candidate_counts arrays
    return Candidates();
}

Candidates query_vectors(cublasHandle_t cublas_handle, cudaStream_t stream, const double* Q,
                         int n_queries, int n_features, const RPLSHParams& params, const double* P,
                         const Index* index, const double* X_stored) {

    // TODO: Hash query vectors using projections (double precision)
    // TODO: Query index for candidate indices (double precision)
    // TODO: Retrieve actual vectors from X_stored using indices (double precision)
    // TODO: Return results in candidate_vectors and candidate_counts arrays
    return Candidates();
}

} // namespace rplsh
} // namespace culsh
