#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>

#ifndef NDEBUG
#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t error = call;                                                                  \
        if (error != cudaSuccess) {                                                                \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__,                      \
                    cudaGetErrorString(error));                                                    \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)

#define CURAND_CHECK(call)                                                                         \
    do {                                                                                           \
        curandStatus_t status = call;                                                              \
        if (status != CURAND_STATUS_SUCCESS) {                                                     \
            fprintf(stderr, "cuRAND error at %s:%d - %d\n", __FILE__, __LINE__, status);           \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

#define CUBLAS_CHECK(call)                                                                         \
    do {                                                                                           \
        cublasStatus_t status = call;                                                              \
        if (status != CUBLAS_STATUS_SUCCESS) {                                                     \
            fprintf(stderr, "cuBLAS error at %s:%d - %d\n", __FILE__, __LINE__, status);           \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)
#else
#define CUDA_CHECK(call) call
#define CURAND_CHECK(call) call
#define CUBLAS_CHECK(call) call
#endif
