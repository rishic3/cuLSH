#include "rplsh/index.cuh"
#include "rplsh/kernels/build_index.cuh"
#include "rplsh/kernels/hash.cuh"
#include "rplsh/kernels/projections.cuh"
#include "rplsh/kernels/query_index.cuh"

#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <curand.h>
#include <iomanip>
#include <iostream>
#include <vector>

struct Params {
    int n_hash_tables;
    int n_projections;
    int seed;
};

void generate_x(float* X, int n, int d, int seed) {
    curandGenerator_t rng;
    CURAND_CHECK(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(rng, seed));

    size_t x_size = static_cast<size_t>(n) * d;
    CURAND_CHECK(curandGenerateUniform(rng, X, x_size));

    CURAND_CHECK(curandDestroyGenerator(rng));
}

culsh::rplsh::Index main_fit(cublasHandle_t cublas_handle, cudaStream_t stream, const float* X,
                             int n_samples, int n_features, const Params& params, float* P) {

    // allocate X_hash
    float* X_hash;
    CUDA_CHECK(cudaMalloc(&X_hash, static_cast<size_t>(n_samples) * params.n_hash_tables *
                                       params.n_projections * sizeof(float)));

    // generate random projections and hash X
    culsh::rplsh::detail::generate_random_projections<float>(
        stream, params.n_hash_tables * params.n_projections, n_features, params.seed, P);
    culsh::rplsh::detail::hash<float>(cublas_handle, stream, X, P, n_samples, n_features,
                                      params.n_hash_tables, params.n_projections, X_hash);

    // build index directly from hash values (no separate signature step needed)
    auto index = culsh::rplsh::detail::build_index(stream, X_hash, n_samples, params.n_hash_tables,
                                                   params.n_projections);
    CUDA_CHECK(cudaFree(X_hash));

    return index;
}

void test() {
    const int n = 1000000;
    const int d = 128;
    const int n_hash_tables = 64;
    const int n_projections = 8;
    const int n_total_projections = n_hash_tables * n_projections;

    // create stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // create cublas handle
    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));

    float* X;
    float* P;

    // allocate memory for X, P
    CUDA_CHECK(cudaMalloc(&X, static_cast<size_t>(n) * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&P, static_cast<size_t>(n_total_projections) * d * sizeof(float)));

    generate_x(X, n, d, 12345);

    auto params = Params{n_hash_tables, n_projections, 12345};

    auto start_fit = std::chrono::high_resolution_clock::now();
    // Fit the LSH model and get the index
    culsh::rplsh::Index index = main_fit(cublas_handle, stream, X, n, d, params, P);
    auto end_fit = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> fit_time = end_fit - start_fit;

    std::cout << "Built index with " << index.n_hash_tables << " hash tables and "
              << index.n_projections << " projections per table" << std::endl;
    std::cout << "  Fit took [" << fit_time.count() << "s]" << std::endl;

    CUDA_CHECK(cudaFree(X));

    // QUERY
    const int n_queries = 1000;
    float* Q;
    CUDA_CHECK(cudaMalloc(&Q, static_cast<size_t>(n_queries) * d * sizeof(float)));
    generate_x(Q, n_queries, d, 42);

    float* Q_hash;
    CUDA_CHECK(
        cudaMalloc(&Q_hash, static_cast<size_t>(n_queries) * n_total_projections * sizeof(float)));

    auto start_query = std::chrono::high_resolution_clock::now();

    culsh::rplsh::detail::hash<float>(cublas_handle, stream, Q, P, n_queries, d, n_hash_tables,
                                      n_projections, Q_hash);

    CUDA_CHECK(cudaFree(P));
    CUDA_CHECK(cudaFree(Q));

    // query index directly with hash values
    culsh::rplsh::Candidates candidates = culsh::rplsh::detail::query_index(
        stream, Q_hash, n_queries, n_hash_tables, n_projections, &index);

    auto end_query = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> query_time = end_query - start_query;
    std::cout << "  Query took [" << query_time.count() << "s]" << std::endl;

    CUDA_CHECK(cudaFree(Q_hash));
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

void test_breakdown() {
    const int n = 1000000;
    const int d = 128;
    const int n_hash_tables = 64;
    const int n_projections = 8;
    const int n_total_projections = n_hash_tables * n_projections;

    float* X;
    float* P;
    float* X_hash;

    // create stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // create cublas handle
    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));

    // allocate memory for X, P, X_hash
    CUDA_CHECK(cudaMalloc(&X, static_cast<size_t>(n) * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&P, static_cast<size_t>(n_total_projections) * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&X_hash, static_cast<size_t>(n) * n_total_projections * sizeof(float)));

    // generate X
    generate_x(X, n, d, 12345);

    auto start_generate_projections = std::chrono::high_resolution_clock::now();
    // generate random projections
    culsh::rplsh::detail::generate_random_projections<float>(stream, n_total_projections, d, 12345,
                                                             P);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto end_generate_projections = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> generate_projections_time =
        end_generate_projections - start_generate_projections;
    std::cout << "Generated projections in " << generate_projections_time.count() << " sec"
              << std::endl;

    auto start_hash = std::chrono::high_resolution_clock::now();
    // hash
    culsh::rplsh::detail::hash<float>(cublas_handle, stream, X, P, n, d, n_hash_tables,
                                      n_projections, X_hash);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto end_hash = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> hash_time = end_hash - start_hash;
    std::cout << "Hashed in " << hash_time.count() << " sec" << std::endl;

    // free input and projections
    CUDA_CHECK(cudaFree(P));
    CUDA_CHECK(cudaFree(X));

    auto start_build_index = std::chrono::high_resolution_clock::now();
    // build index directly from hash values
    culsh::rplsh::Index index =
        culsh::rplsh::detail::build_index(stream, X_hash, n, n_hash_tables, n_projections);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto end_build_index = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> build_index_time = end_build_index - start_build_index;
    std::cout << "Built index in " << build_index_time.count() << " sec" << std::endl;

    std::chrono::duration<double> total_time = end_build_index - start_generate_projections;
    std::cout << "   Total fit time in " << total_time.count() << " sec" << std::endl;

    // print index metadata
    std::cout << "Index metadata: " << std::endl;
    std::cout << "  n_total_buckets: " << index.n_total_buckets << std::endl;
    std::cout << "  n_hash_tables: " << index.n_hash_tables << std::endl;
    std::cout << "  n_projections: " << index.n_projections << std::endl;

    CUDA_CHECK(cudaGetLastError());

    // free hash values
    CUDA_CHECK(cudaFree(X_hash));
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Using GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB\n" << std::endl;

    test();
    return 0;
}
