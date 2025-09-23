#include "rplsh/index.cuh"
#include "rplsh/kernels/hash.cuh"
#include "rplsh/kernels/indexing.cuh"
#include "rplsh/kernels/projections.cuh"

#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <curand.h>
#include <iomanip>
#include <iostream>
#include <vector>

void generate_x(float* X, int n, int d, int seed) {
    curandGenerator_t rng;
    CURAND_CHECK(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(rng, seed));

    size_t x_size = static_cast<size_t>(n) * d;
    CURAND_CHECK(curandGenerateUniform(rng, X, x_size));

    CURAND_CHECK(curandDestroyGenerator(rng));
}

void test_generate_random_projections(bool validate) {
    const int n = 10000000;
    const int d = 128;

    size_t projection_size = static_cast<size_t>(n) * d;
    float* projections;

    // create stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // allocate projections on device
    CUDA_CHECK(cudaMalloc(&projections, projection_size * sizeof(float)));

    cudaDeviceProp device_prop;
    CUDA_CHECK(cudaGetDeviceProperties(&device_prop, 0));

    auto start_time = std::chrono::high_resolution_clock::now();

    // generate random projections
    culsh::rplsh::detail::generate_random_projections<float>(stream, n, d, 12345, projections);
    // culsh::rplsh::detail::generate_random_projections_two_kernel<float>(stream, n, d, 12345,
    // device_prop, projections);

    CUDA_CHECK(cudaGetLastError());

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> kernel_time = end_time - start_time;
    std::cout << "Completed in " << kernel_time.count() << " sec" << std::endl;

    std::cout << "Generated " << n << " random projection vectors of dimension " << d << std::endl;

    if (validate) {
        std::vector<float> h_P(projection_size);
        CUDA_CHECK(cudaMemcpy(h_P.data(), projections, projection_size, cudaMemcpyDeviceToHost));

        // Check first few rows to see if normalization worked
        std::cout << "\nFirst 3 projection vectors:" << std::endl;
        for (int i = 0; i < std::min(3, n); i++) {
            std::cout << "Vector " << i << ": ";

            // Print first few elements of this row
            for (int j = 0; j < std::min(5, d); j++) {
                std::cout << std::setw(8) << std::fixed << std::setprecision(4) << h_P[i * d + j]
                          << " ";
            }
            if (d > 5)
                std::cout << "...";
            std::cout << std::endl;
        }

        // Verify normalization: each row should have unit norm
        std::cout << "\nVerifying normalization (each vector should have norm ≈ 1.0):" << std::endl;

        for (int i = 0; i < n; i++) {
            float norm_sq = 0.0f;

            // Compute norm of row i
            for (int j = 0; j < d; j++) {
                float val = h_P[i * d + j];
                norm_sq += val * val;
            }

            float norm = std::sqrt(norm_sq);

            // Print first few norms
            if (i < 5) {
                std::cout << "Vector " << i << " norm: " << std::setprecision(6) << norm
                          << std::endl;
            }

            // Check if norm is approximately 1.0
            if (std::abs(norm - 1.0f) > 1e-8f) {
                std::cout << "Vector " << i << " is not normalized" << std::endl;
            }
        }
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(projections));
}

void test_hash() {
    const int n = 10000000;
    const int d = 128;
    const int n_hash_tables = 64;
    const int n_projections = 8;
    const int n_total_buckets = n_hash_tables * n_projections;

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
    CUDA_CHECK(cudaMalloc(&P, static_cast<size_t>(n_total_buckets) * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&X_hash, static_cast<size_t>(n) * n_total_buckets * sizeof(float)));

    // generate X
    generate_x(X, n, d, 12345);

    auto start_generate_projections = std::chrono::high_resolution_clock::now();
    // generate random projections
    culsh::rplsh::detail::generate_random_projections<float>(stream, n_total_buckets, d, 12345, P);
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

    // free projections
    CUDA_CHECK(cudaFree(P));
    // free X
    CUDA_CHECK(cudaFree(X));

    // allocate memory for signatures
    int8_t* X_signatures;
    CUDA_CHECK(
        cudaMalloc(&X_signatures, static_cast<size_t>(n) * n_total_buckets * sizeof(int8_t)));

    auto start_compute_signatures = std::chrono::high_resolution_clock::now();
    // convert hash values to signatures
    culsh::rplsh::detail::compute_signatures<float>(stream, X_hash, n, n_hash_tables, n_projections,
                                                    X_signatures);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto end_compute_signatures = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> compute_signatures_time =
        end_compute_signatures - start_compute_signatures;
    std::cout << "Computed signatures in " << compute_signatures_time.count() << " sec"
              << std::endl;

    auto start_build_index = std::chrono::high_resolution_clock::now();
    // build index
    culsh::rplsh::RPLSHIndex index =
        culsh::rplsh::detail::build_index(stream, X_signatures, n, n_hash_tables, n_projections);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto end_build_index = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> build_index_time = end_build_index - start_build_index;
    std::cout << "Built index in " << build_index_time.count() << " sec" << std::endl;

    // print index metadata
    std::cout << "Index metadata: " << std::endl;
    std::cout << "  n_total_buckets: " << index.n_total_buckets << std::endl;
    std::cout << "  n_hash_tables: " << index.n_hash_tables << std::endl;
    std::cout << "  n_projections: " << index.n_projections << std::endl;

    CUDA_CHECK(cudaGetLastError());

    // free signatures
    CUDA_CHECK(cudaFree(X_signatures));
    CUDA_CHECK(cudaFree(X_hash));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Using GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB\n" << std::endl;

    // test_generate_random_projections(false);
    test_hash();
    return 0;
}
