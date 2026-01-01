#pragma once

#include "../core/utils.cuh"
#include <cstdint>
#include <cub/block/block_reduce.cuh>
#include <cuda_runtime.h>
#include <curand.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdexcept>

namespace culsh {
namespace rplsh {
namespace detail {

/**
 * @brief Sample n random vectors from a d-dimensional sphere
 * @param[in] stream CUDA stream
 * @param[in] n Number of vectors to generate
 * @param[in] d Dimensionality of each vector
 * @param[in] seed Seed for the random number generator
 * @param[out] P Device pointer to n x d array of random vectors (float or double)
 */
template <typename DType>
void generate_random_projections(cudaStream_t stream, int n, int d, uint64_t seed, DType* P) {
    curandGenerator_t rng;
    CURAND_CHECK(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetStream(rng, stream));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(rng, seed));

    size_t projection_size = static_cast<size_t>(n) * d;

    if constexpr (std::is_same_v<DType, float>) {
        // Generate random normal floats
        CURAND_CHECK(curandGenerateNormal(rng, P, projection_size, 0.0f, 1.0f));
    } else if constexpr (std::is_same_v<DType, double>) {
        // Generate random normal doubles
        CURAND_CHECK(curandGenerateNormalDouble(rng, P, projection_size, 0.0, 1.0));
    } else {
        throw std::invalid_argument("Expected float or double type");
    }

    CURAND_CHECK(curandDestroyGenerator(rng));
}

} // namespace detail
} // namespace rplsh
} // namespace culsh
