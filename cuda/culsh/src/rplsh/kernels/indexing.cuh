#pragma once

#include "../utils/utils.cuh"

namespace culsh {
namespace rplsh {
namespace detail {

// TODO: create index kernel

template <typename DType>
void create_index_impl(cudaStream_t stream, int n_hash_tables, int n_projections, int n_cols,
                       const DType* X_hash) {}

// TODO: query index kernel

} // namespace detail
} // namespace rplsh
} // namespace culsh
