#pragma once

#include "../utils/utils.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <chrono>
#include <iostream>

namespace culsh {
namespace rplsh {

__global__ void normalize_rows_kernel(float* P, int n_rows, int n_cols) {
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_idx < n_rows) {
        // compute row-wise norm
        float row_norm = 0.0f;
        for (int col_idx = 0; col_idx < n_cols; ++col_idx) {
            float val = P[row_idx * n_cols + col_idx];
            row_norm += val * val;
        }
        row_norm = sqrtf(row_norm);

        // normalize row
        if (row_norm > 1e-8f) {  // check div by zero
            for (int col_idx = 0; col_idx < n_cols; ++col_idx) {
                P[row_idx * n_cols + col_idx] /= row_norm;
            }
        }
    }
}

__global__ void calculate_norms_kernel(const float* P, float* norms, int n_rows, int n_cols) {
    // Each block calculates the norm for one row
    int row = blockIdx.x;
    if (row >= n_rows) return;

    // Use shared memory for parallel reduction within a block
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    float sum_sq = 0.0f;
    // Each thread calculates a partial sum of squares
    for (unsigned int i = tid; i < n_cols; i += blockDim.x) {
        float val = P[row * n_cols + i];
        sum_sq += val * val;
    }
    sdata[tid] = sum_sq;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // First thread writes the final result (sqrt of sum of squares)
    if (tid == 0) {
        norms[row] = sqrtf(sdata[0]);
    }
}

__global__ void normalize_vectors_kernel(float* P, const float* norms, int n_rows, int n_cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n_rows && col < n_cols) {
        float norm = norms[row];
        // Avoid division by zero
        if (norm > 1e-8f) {
            P[row * n_cols + col] /= norm;
        }
    }
}

extern "C" {

float* generate_random_projections(int n, int d) {
    // create curand generator
    curandGenerator_t rng;
    CURAND_CHECK(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT));

    // allocate space for output matrix on device
    float* P;
    size_t projection_size = static_cast<size_t>(n) * d;

    auto start_time = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaMalloc(&P, projection_size * sizeof(float)));

    // generate n_hash * d random floats
    CURAND_CHECK(curandGenerateNormal(rng, P, projection_size, 0.0f, 1.0f));

    // launch row-wise norm kernel
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    normalize_rows_kernel<<<grid_size, block_size>>>(P, n, d);

    CUDA_CHECK(cudaDeviceSynchronize());

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> kernel_time = end_time - start_time;
    std::cout << "Single kernel approach completed in " << kernel_time.count() << " sec" << std::endl;

    CURAND_CHECK(curandDestroyGenerator(rng));

    return P;
}

float* generate_random_projections_two_kernels(int n, int d) {
    curandGenerator_t rng;
    CURAND_CHECK(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT));

    float* P;
    float* norms;
    size_t projection_size = static_cast<size_t>(n) * d;
    size_t norms_size = static_cast<size_t>(n);

    auto start_time = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaMalloc(&P, projection_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&norms, norms_size * sizeof(float)));

    CURAND_CHECK(curandGenerateNormal(rng, P, projection_size, 0.0f, 1.0f));

    calculate_norms_kernel<<<n, 256, 256 * sizeof(float)>>>(P, norms, n, d);
    CUDA_CHECK(cudaGetLastError());
    
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((d + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (n + threadsPerBlock.y - 1) / threadsPerBlock.y);
    normalize_vectors_kernel<<<numBlocks, threadsPerBlock>>>(P, norms, n, d);

    CUDA_CHECK(cudaDeviceSynchronize());

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> kernel_time = end_time - start_time;
    std::cout << "Two kernel approach completed in " << kernel_time.count() << " sec" << std::endl;

    CURAND_CHECK(curandDestroyGenerator(rng));
    CUDA_CHECK(cudaFree(norms));

    return P;
}

} // extern "C"

} // namespace rplsh
} // namespace culsh
