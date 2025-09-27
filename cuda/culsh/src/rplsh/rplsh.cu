#include "index.cuh"
#include "kernels/hash.cuh"
#include "kernels/indexing.cuh"
#include "kernels/projections.cuh"
#include <cuda_runtime.h>
#include <culsh/rplsh/params.hpp>
#include <culsh/rplsh/rplsh.hpp>
#include <curand.h>

namespace culsh {
namespace rplsh {

RPLSHIndex fit(cublasHandle_t cublas_handle, cudaStream_t stream, const float* X, int n_samples,
               int n_features, const RPLSHParams& params, float* P) {

    // allocate X_hash
    float* X_hash;
    CUDA_CHECK(cudaMalloc(&X_hash, static_cast<size_t>(n_samples) * params.n_hash_tables *
                                       params.n_projections * sizeof(float)));

    // generate random projections and hash X
    const int n_items = n_samples * params.n_hash_tables;
    detail::generate_random_projections<float>(stream, n_items, n_features, params.seed, P);
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

RPLSHIndex fit(cublasHandle_t cublas_handle, cudaStream_t stream, const double* X, int n_samples,
               int n_features, const RPLSHParams& params, double* P) {

    // allocate X_hash
    double* X_hash;
    CUDA_CHECK(cudaMalloc(&X_hash, static_cast<size_t>(n_samples) * params.n_hash_tables *
                                       params.n_projections * sizeof(double)));

    // generate random projections and hash X
    const int n_items = n_samples * params.n_hash_tables;
    detail::generate_random_projections<double>(stream, n_items, n_features, params.seed, P);
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

void query_indices(cublasHandle_t cublas_handle, cudaStream_t stream, const float* Q, int n_queries,
                   int n_features, const RPLSHParams& params, const float* P,
                   const RPLSHIndex* index, int* candidates, int* candidate_counts) {

    // TODO: Hash query vectors using projections
    // TODO: Query index for candidate indices
    // TODO: Return results in candidates and candidate_counts arrays
}

void query_indices(cublasHandle_t cublas_handle, cudaStream_t stream, const double* Q,
                   int n_queries, int n_features, const RPLSHParams& params, const double* P,
                   const RPLSHIndex* index, int* candidates, int* candidate_counts) {

    // TODO: Hash query vectors using projections (double precision)
    // TODO: Query index for candidate indices (double precision)
    // TODO: Return results in candidates and candidate_counts arrays
}

void query_vectors(cublasHandle_t cublas_handle, cudaStream_t stream, const float* Q, int n_queries,
                   int n_features, const RPLSHParams& params, const float* P,
                   const RPLSHIndex* index, const float* X_stored, float* candidate_vectors,
                   int* candidate_counts) {

    // TODO: Hash query vectors using projections
    // TODO: Query index for candidate indices
    // TODO: Retrieve actual vectors from X_stored using indices
    // TODO: Return results in candidate_vectors and candidate_counts arrays
}

void query_vectors(cublasHandle_t cublas_handle, cudaStream_t stream, const double* Q,
                   int n_queries, int n_features, const RPLSHParams& params, const double* P,
                   const RPLSHIndex* index, const double* X_stored, double* candidate_vectors,
                   int* candidate_counts) {

    // TODO: Hash query vectors using projections (double precision)
    // TODO: Query index for candidate indices (double precision)
    // TODO: Retrieve actual vectors from X_stored using indices (double precision)
    // TODO: Return results in candidate_vectors and candidate_counts arrays
}

} // namespace rplsh
} // namespace culsh
