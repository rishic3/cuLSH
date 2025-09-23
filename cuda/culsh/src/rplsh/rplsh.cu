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

void fit(cublasHandle_t cublas_handle, cudaStream_t stream, const float* X, int n_samples,
         int n_features, const RPLSHParams& params, float* projections, RPLSHIndex* index) {

    // TODO: Generate random projections using templated function

    // TODO: create index
}

void fit(cublasHandle_t cublas_handle, cudaStream_t stream, const double* X, int n_samples,
         int n_features, const RPLSHParams& params, double* projections, RPLSHIndex* index) {

    // TODO: Generate random projections using templated function

    // TODO: create index
}

void query_indices(cublasHandle_t cublas_handle, cudaStream_t stream, const float* Q, int n_queries,
                   int n_features, const RPLSHParams& params, const float* projections,
                   const RPLSHIndex* index, int* candidates, int* candidate_counts) {

    // TODO: Hash query vectors using projections
    // TODO: Query index for candidate indices
    // TODO: Return results in candidates and candidate_counts arrays
}

void query_indices(cublasHandle_t cublas_handle, cudaStream_t stream, const double* Q,
                   int n_queries, int n_features, const RPLSHParams& params,
                   const double* projections, const RPLSHIndex* index, int* candidates,
                   int* candidate_counts) {

    // TODO: Hash query vectors using projections (double precision)
    // TODO: Query index for candidate indices (double precision)
    // TODO: Return results in candidates and candidate_counts arrays
}

void query_vectors(cublasHandle_t cublas_handle, cudaStream_t stream, const float* Q, int n_queries,
                   int n_features, const RPLSHParams& params, const float* projections,
                   const RPLSHIndex* index, const float* X_stored, float* candidate_vectors,
                   int* candidate_counts) {

    // TODO: Hash query vectors using projections
    // TODO: Query index for candidate indices
    // TODO: Retrieve actual vectors from X_stored using indices
    // TODO: Return results in candidate_vectors and candidate_counts arrays
}

void query_vectors(cublasHandle_t cublas_handle, cudaStream_t stream, const double* Q,
                   int n_queries, int n_features, const RPLSHParams& params,
                   const double* projections, const RPLSHIndex* index, const double* X_stored,
                   double* candidate_vectors, int* candidate_counts) {

    // TODO: Hash query vectors using projections (double precision)
    // TODO: Query index for candidate indices (double precision)
    // TODO: Retrieve actual vectors from X_stored using indices (double precision)
    // TODO: Return results in candidate_vectors and candidate_counts arrays
}

} // namespace rplsh
} // namespace culsh
