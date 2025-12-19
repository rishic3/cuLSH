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
#include <thrust/execution_policy.h>
#include <thrust/transform.h>

namespace culsh {
namespace rplsh {

namespace {
// Functor to convert double to float
struct DoubleToFloatFunctor {
    __host__ __device__ float operator()(double x) const { return static_cast<float>(x); }
};
} // anonymous namespace

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

    // build index directly from hash values
    auto index =
        detail::build_index(stream, X_hash, n_samples, params.n_hash_tables, params.n_projections);
    CUDA_CHECK(cudaFree(X_hash));

    return index;
}

Index fit(cublasHandle_t cublas_handle, cudaStream_t stream, const double* X, int n_samples,
          int n_features, const RPLSHParams& params, double* P) {

    // allocate X_hash (as float for index building)
    double* X_hash_double;
    float* X_hash;
    size_t hash_size = static_cast<size_t>(n_samples) * params.n_hash_tables * params.n_projections;

    CUDA_CHECK(cudaMalloc(&X_hash_double, hash_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&X_hash, hash_size * sizeof(float)));

    // generate random projections and hash X
    detail::generate_random_projections<double>(stream, params.n_hash_tables * params.n_projections,
                                                n_features, params.seed, P);
    detail::hash<double>(cublas_handle, stream, X, P, n_samples, n_features, params.n_hash_tables,
                         params.n_projections, X_hash_double);

    // Convert double hash to float for index building
    // (only sign matters for signature, so float precision is sufficient)
    thrust::transform(thrust::cuda::par.on(stream), X_hash_double, X_hash_double + hash_size,
                      X_hash, DoubleToFloatFunctor());
    CUDA_CHECK(cudaFree(X_hash_double));

    // build index from float hash values
    auto index =
        detail::build_index(stream, X_hash, n_samples, params.n_hash_tables, params.n_projections);
    CUDA_CHECK(cudaFree(X_hash));

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

    // query index directly with hash values
    auto candidates = detail::query_index(stream, Q_hash, n_queries, params.n_hash_tables,
                                          params.n_projections, index);
    CUDA_CHECK(cudaFree(Q_hash));

    return candidates;
}

Candidates query_indices(cublasHandle_t cublas_handle, cudaStream_t stream, const double* Q,
                         int n_queries, int n_features, const RPLSHParams& params, const double* P,
                         const Index* index) {

    // allocate Q_hash (as float for querying)
    double* Q_hash_double;
    float* Q_hash;
    size_t hash_size = static_cast<size_t>(n_queries) * params.n_hash_tables * params.n_projections;

    CUDA_CHECK(cudaMalloc(&Q_hash_double, hash_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&Q_hash, hash_size * sizeof(float)));

    detail::hash<double>(cublas_handle, stream, Q, P, n_queries, n_features, params.n_hash_tables,
                         params.n_projections, Q_hash_double);

    // Convert double hash to float for querying
    thrust::transform(thrust::cuda::par.on(stream), Q_hash_double, Q_hash_double + hash_size,
                      Q_hash, DoubleToFloatFunctor());
    CUDA_CHECK(cudaFree(Q_hash_double));

    // query index with float hash values
    auto candidates = detail::query_index(stream, Q_hash, n_queries, params.n_hash_tables,
                                          params.n_projections, index);
    CUDA_CHECK(cudaFree(Q_hash));

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
