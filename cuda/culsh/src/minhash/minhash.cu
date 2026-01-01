#include "../core/candidates.cuh"
#include "../core/fit.cuh"
#include "../core/fit_query.cuh"
#include "../core/index.cuh"
#include "../core/query.cuh"
#include "hash.cuh"
#include "index.cuh"
#include <cassert>
#include <cuda_runtime.h>
#include <culsh/minhash/minhash.hpp>
#include <culsh/minhash/params.hpp>
#include <curand.h>

namespace culsh {
namespace minhash {

using Candidates = core::Candidates;

Index fit(cudaStream_t stream, const int* X_indices, const int* X_indptr, int n_samples,
          int n_features, const MinHashParams& params) {
    // Allocate hash integers
    int total_hashes = params.n_hash_tables * params.n_hashes;
    uint32_t* A = nullptr;
    uint32_t* B = nullptr;
    CUDA_CHECK(cudaMalloc(&A, total_hashes * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&B, total_hashes * sizeof(uint32_t)));
    detail::generate_hash_integers(stream, total_hashes, params.seed, A, B);

    // Allocate X_sig
    uint32_t* X_sig = nullptr;
    CUDA_CHECK(cudaMalloc(&X_sig, static_cast<size_t>(n_samples) * params.n_hash_tables *
                                      params.n_hashes * sizeof(uint32_t)));
    detail::compute_minhash(stream, X_indices, X_indptr, A, B, n_samples, params.n_hash_tables,
                            params.n_hashes, X_sig);

    // Build index
    int sig_nbytes = params.n_hashes * static_cast<int>(sizeof(uint32_t));
    core::Index core_index = core::detail::fit_index(stream, X_sig, n_samples, params.n_hash_tables,
                                                     params.n_hashes, sig_nbytes);
    CUDA_CHECK(cudaFree(X_sig));

    // Wrap core index and store hash integers + metadata
    Index index;
    index.core = std::move(core_index);
    index.core.n_features = n_features;
    index.core.seed = params.seed;
    index.A = A;
    index.B = B;
    return index;
}

Candidates query(cudaStream_t stream, const int* Q_indices, const int* Q_indptr, int n_queries,
                 const Index& index) {
    int n_hash_tables = index.core.n_hash_tables;
    int n_hashes = index.core.n_hashes;
    int sig_nbytes = index.core.sig_nbytes;

    // Allocate Q_sig
    uint32_t* Q_sig = nullptr;
    CUDA_CHECK(cudaMalloc(&Q_sig, static_cast<size_t>(n_queries) * n_hash_tables * n_hashes *
                                      sizeof(uint32_t)));

    // Compute minhash signatures
    detail::compute_minhash(stream, Q_indices, Q_indptr, index.A, index.B, n_queries, n_hash_tables,
                            n_hashes, Q_sig);

    // Query index
    Candidates candidates =
        core::detail::query_index(stream, Q_sig, n_queries, sig_nbytes, &index.core);
    CUDA_CHECK(cudaFree(Q_sig));

    return candidates;
}

Candidates fit_query(cudaStream_t stream, const int* X_indices, const int* X_indptr, int n_samples,
                     int n_features, const MinHashParams& params) {
    // Allocate hash integers
    int total_hashes = params.n_hash_tables * params.n_hashes;
    uint32_t* A = nullptr;
    uint32_t* B = nullptr;
    CUDA_CHECK(cudaMalloc(&A, total_hashes * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&B, total_hashes * sizeof(uint32_t)));
    detail::generate_hash_integers(stream, total_hashes, params.seed, A, B);

    // Allocate X_sig
    uint32_t* X_sig = nullptr;
    CUDA_CHECK(cudaMalloc(&X_sig, static_cast<size_t>(n_samples) * params.n_hash_tables *
                                      params.n_hashes * sizeof(uint32_t)));
    detail::compute_minhash(stream, X_indices, X_indptr, A, B, n_samples, params.n_hash_tables,
                            params.n_hashes, X_sig);

    // Fit index and query candidates directly (X_sig == Q_sig)
    int sig_nbytes = params.n_hashes * static_cast<int>(sizeof(uint32_t));
    Candidates candidates = core::detail::fit_query(stream, X_sig, n_samples, params.n_hash_tables,
                                                    params.n_hashes, sig_nbytes);

    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(X_sig));

    return candidates;
}

} // namespace minhash
} // namespace culsh
