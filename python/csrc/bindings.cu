#include "common/array_utils.cuh"
#include "common/cuda_utils.cuh"

#include <core/candidates.cuh>
#include <core/index.cuh>
#include <culsh/minhash/minhash.hpp>
#include <culsh/minhash/params.hpp>
#include <culsh/pslsh/params.hpp>
#include <culsh/pslsh/pslsh.hpp>
#include <culsh/rplsh/params.hpp>
#include <culsh/rplsh/rplsh.hpp>
#include <minhash/index.cuh>
#include <pslsh/index.cuh>
#include <rplsh/index.cuh>

#include <optional>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace culsh {
namespace python {

// --- RPLSHCore ---
class RPLSHCore : public CUDAResourceManager {
private:
    int n_hash_tables_ = 0;
    int n_hashes_ = 0;
    uint64_t seed_ = 0;

public:
    RPLSHCore(int n_hash_tables, int n_hashes, uint64_t seed = 42)
        : CUDAResourceManager(), n_hash_tables_(n_hash_tables), n_hashes_(n_hashes), seed_(seed) {}

    std::unique_ptr<rplsh::Index> fit_float(py::object X_obj, int n_samples, int n_features) {
        float* X_ptr = get_device_pointer<float>(X_obj);

        culsh::RPLSHParams params{n_hash_tables_, n_hashes_, seed_};
        auto index = rplsh::fit(cublas_handle_, stream_, X_ptr, n_samples, n_features, params);

        synchronize();
        return std::make_unique<rplsh::Index>(std::move(index));
    }

    std::unique_ptr<rplsh::Index> fit_double(py::object X_obj, int n_samples, int n_features) {
        double* X_ptr = get_device_pointer<double>(X_obj);

        culsh::RPLSHParams params{n_hash_tables_, n_hashes_, seed_};
        auto index = rplsh::fit(cublas_handle_, stream_, X_ptr, n_samples, n_features, params);

        synchronize();
        return std::make_unique<rplsh::Index>(std::move(index));
    }

    std::unique_ptr<rplsh::Candidates> query_float(py::object Q_obj, int n_queries,
                                                   const rplsh::Index& index,
                                                   std::optional<int> batch_size = std::nullopt) {
        if (index.is_double) {
            throw std::runtime_error("Index was fitted with float64, but query is float32");
        }

        float* Q_ptr = get_device_pointer<float>(Q_obj);
        auto candidates = batch_size.has_value()
                              ? rplsh::query_batched(cublas_handle_, stream_, Q_ptr, n_queries,
                                                     index, batch_size.value())
                              : rplsh::query(cublas_handle_, stream_, Q_ptr, n_queries, index);

        synchronize();
        return std::make_unique<rplsh::Candidates>(std::move(candidates));
    }

    std::unique_ptr<rplsh::Candidates> query_double(py::object Q_obj, int n_queries,
                                                    const rplsh::Index& index,
                                                    std::optional<int> batch_size = std::nullopt) {
        if (!index.is_double) {
            throw std::runtime_error("Index was fitted with float32, but query is float64");
        }

        double* Q_ptr = get_device_pointer<double>(Q_obj);
        auto candidates = batch_size.has_value()
                              ? rplsh::query_batched(cublas_handle_, stream_, Q_ptr, n_queries,
                                                     index, batch_size.value())
                              : rplsh::query(cublas_handle_, stream_, Q_ptr, n_queries, index);

        synchronize();
        return std::make_unique<rplsh::Candidates>(std::move(candidates));
    }

    std::unique_ptr<rplsh::Candidates> fit_query_float(py::object X_obj, int n_samples,
                                                       int n_features) {
        float* X_ptr = get_device_pointer<float>(X_obj);

        culsh::RPLSHParams params{n_hash_tables_, n_hashes_, seed_};
        auto candidates =
            rplsh::fit_query(cublas_handle_, stream_, X_ptr, n_samples, n_features, params);

        synchronize();
        return std::make_unique<rplsh::Candidates>(std::move(candidates));
    }

    std::unique_ptr<rplsh::Candidates> fit_query_double(py::object X_obj, int n_samples,
                                                        int n_features) {
        double* X_ptr = get_device_pointer<double>(X_obj);

        culsh::RPLSHParams params{n_hash_tables_, n_hashes_, seed_};
        auto candidates =
            rplsh::fit_query(cublas_handle_, stream_, X_ptr, n_samples, n_features, params);

        synchronize();
        return std::make_unique<rplsh::Candidates>(std::move(candidates));
    }

    int n_hash_tables() const { return n_hash_tables_; }
    int n_hashes() const { return n_hashes_; }
    uint64_t seed() const { return seed_; }
};

// --- MinHashCore ---
class MinHashCore : public CUDAResourceManager {
private:
    int n_hash_tables_ = 0;
    int n_hashes_ = 0;
    uint64_t seed_ = 0;

public:
    MinHashCore(int n_hash_tables, int n_hashes, uint64_t seed = 42)
        : CUDAResourceManager(), n_hash_tables_(n_hash_tables), n_hashes_(n_hashes), seed_(seed) {}

    std::unique_ptr<minhash::Index> fit(py::object indices_obj, py::object indptr_obj,
                                        int n_samples, int n_features) {
        int* indices_ptr = get_device_pointer<int>(indices_obj);
        int* indptr_ptr = get_device_pointer<int>(indptr_obj);

        culsh::MinHashParams params{n_hash_tables_, n_hashes_, seed_};
        auto index = minhash::fit(stream_, indices_ptr, indptr_ptr, n_samples, n_features, params);

        synchronize();
        return std::make_unique<minhash::Index>(std::move(index));
    }

    std::unique_ptr<minhash::Candidates> query(py::object indices_obj, py::object indptr_obj,
                                               int n_queries, const minhash::Index& index) {
        int* indices_ptr = get_device_pointer<int>(indices_obj);
        int* indptr_ptr = get_device_pointer<int>(indptr_obj);

        auto candidates = minhash::query(stream_, indices_ptr, indptr_ptr, n_queries, index);

        synchronize();
        return std::make_unique<minhash::Candidates>(std::move(candidates));
    }

    std::unique_ptr<minhash::Candidates> fit_query(py::object indices_obj, py::object indptr_obj,
                                                   int n_samples, int n_features) {
        int* indices_ptr = get_device_pointer<int>(indices_obj);
        int* indptr_ptr = get_device_pointer<int>(indptr_obj);

        culsh::MinHashParams params{n_hash_tables_, n_hashes_, seed_};
        auto candidates =
            minhash::fit_query(stream_, indices_ptr, indptr_ptr, n_samples, n_features, params);

        synchronize();
        return std::make_unique<minhash::Candidates>(std::move(candidates));
    }

    int n_hash_tables() const { return n_hash_tables_; }
    int n_hashes() const { return n_hashes_; }
    uint64_t seed() const { return seed_; }
};

// --- PSLSHCore ---
class PSLSHCore : public CUDAResourceManager {
private:
    int n_hash_tables_ = 0;
    int n_hashes_ = 0;
    int window_size_ = 0;
    uint64_t seed_ = 0;

public:
    PSLSHCore(int n_hash_tables, int n_hashes, int window_size, uint64_t seed = 42)
        : CUDAResourceManager(), n_hash_tables_(n_hash_tables), n_hashes_(n_hashes),
          window_size_(window_size), seed_(seed) {}

    std::unique_ptr<pslsh::Index> fit_float(py::object X_obj, int n_samples, int n_features) {
        float* X_ptr = get_device_pointer<float>(X_obj);

        culsh::PSLSHParams params{n_hash_tables_, n_hashes_, window_size_, seed_};
        auto index = pslsh::fit(cublas_handle_, stream_, X_ptr, n_samples, n_features, params);

        synchronize();
        return std::make_unique<pslsh::Index>(std::move(index));
    }

    std::unique_ptr<pslsh::Index> fit_double(py::object X_obj, int n_samples, int n_features) {
        double* X_ptr = get_device_pointer<double>(X_obj);

        culsh::PSLSHParams params{n_hash_tables_, n_hashes_, window_size_, seed_};
        auto index = pslsh::fit(cublas_handle_, stream_, X_ptr, n_samples, n_features, params);

        synchronize();
        return std::make_unique<pslsh::Index>(std::move(index));
    }

    std::unique_ptr<pslsh::Candidates> query_float(py::object Q_obj, int n_queries,
                                                   const pslsh::Index& index,
                                                   std::optional<int> batch_size = std::nullopt) {
        if (index.is_double) {
            throw std::runtime_error("Index was fitted with float64, but query is float32");
        }

        float* Q_ptr = get_device_pointer<float>(Q_obj);
        auto candidates = batch_size.has_value()
                              ? pslsh::query_batched(cublas_handle_, stream_, Q_ptr, n_queries,
                                                     index, batch_size.value())
                              : pslsh::query(cublas_handle_, stream_, Q_ptr, n_queries, index);

        synchronize();
        return std::make_unique<pslsh::Candidates>(std::move(candidates));
    }

    std::unique_ptr<pslsh::Candidates> query_double(py::object Q_obj, int n_queries,
                                                    const pslsh::Index& index,
                                                    std::optional<int> batch_size = std::nullopt) {
        if (!index.is_double) {
            throw std::runtime_error("Index was fitted with float32, but query is float64");
        }

        double* Q_ptr = get_device_pointer<double>(Q_obj);
        auto candidates = batch_size.has_value()
                              ? pslsh::query_batched(cublas_handle_, stream_, Q_ptr, n_queries,
                                                     index, batch_size.value())
                              : pslsh::query(cublas_handle_, stream_, Q_ptr, n_queries, index);

        synchronize();
        return std::make_unique<pslsh::Candidates>(std::move(candidates));
    }

    std::unique_ptr<pslsh::Candidates> fit_query_float(py::object X_obj, int n_samples,
                                                       int n_features) {
        float* X_ptr = get_device_pointer<float>(X_obj);

        culsh::PSLSHParams params{n_hash_tables_, n_hashes_, window_size_, seed_};
        auto candidates =
            pslsh::fit_query(cublas_handle_, stream_, X_ptr, n_samples, n_features, params);

        synchronize();
        return std::make_unique<pslsh::Candidates>(std::move(candidates));
    }

    std::unique_ptr<pslsh::Candidates> fit_query_double(py::object X_obj, int n_samples,
                                                        int n_features) {
        double* X_ptr = get_device_pointer<double>(X_obj);

        culsh::PSLSHParams params{n_hash_tables_, n_hashes_, window_size_, seed_};
        auto candidates =
            pslsh::fit_query(cublas_handle_, stream_, X_ptr, n_samples, n_features, params);

        synchronize();
        return std::make_unique<pslsh::Candidates>(std::move(candidates));
    }

    int n_hash_tables() const { return n_hash_tables_; }
    int n_hashes() const { return n_hashes_; }
    int window_size() const { return window_size_; }
    uint64_t seed() const { return seed_; }
};

// --- Candidates getters ---
py::object get_candidate_indices(const core::Candidates& candidates, bool as_cupy = false) {
    size_t n = candidates.n_total_candidates;

    if (as_cupy) {
        py::module_ cp = py::module_::import("cupy");
        if (candidates.empty()) {
            return cp.attr("empty")(0, py::arg("dtype") = "int32");
        }
        py::object arr = cp.attr("empty")(n, py::arg("dtype") = "int32");
        int* dst_ptr = get_device_pointer<int>(arr);
        CUDA_CHECK_THROW(cudaMemcpy(dst_ptr, candidates.query_candidate_indices, n * sizeof(int),
                                    cudaMemcpyDeviceToDevice));
        return arr;
    }

    if (candidates.empty()) {
        return py::array_t<int>(0);
    }
    py::array_t<int> result(static_cast<py::ssize_t>(n));
    CUDA_CHECK_THROW(cudaMemcpy(result.mutable_data(), candidates.query_candidate_indices,
                                n * sizeof(int), cudaMemcpyDeviceToHost));
    return result;
}

py::object get_candidate_counts(const core::Candidates& candidates, bool as_cupy = false) {
    size_t n = candidates.n_queries;

    if (as_cupy) {
        py::module_ cp = py::module_::import("cupy");
        if (candidates.empty()) {
            return cp.attr("empty")(0, py::arg("dtype") = "uint64");
        }
        py::object arr = cp.attr("empty")(n, py::arg("dtype") = "uint64");
        size_t* dst_ptr = get_device_pointer<size_t>(arr);
        CUDA_CHECK_THROW(cudaMemcpy(dst_ptr, candidates.query_candidate_counts, n * sizeof(size_t),
                                    cudaMemcpyDeviceToDevice));
        return arr;
    }

    if (candidates.empty()) {
        return py::array_t<size_t>(0);
    }
    py::array_t<size_t> result(static_cast<py::ssize_t>(n));
    CUDA_CHECK_THROW(cudaMemcpy(result.mutable_data(), candidates.query_candidate_counts,
                                n * sizeof(size_t), cudaMemcpyDeviceToHost));
    return result;
}

py::object get_candidate_offsets(const core::Candidates& candidates, bool as_cupy = false) {
    size_t n = candidates.n_queries + 1;

    if (as_cupy) {
        py::module_ cp = py::module_::import("cupy");
        if (candidates.empty()) {
            return cp.attr("empty")(0, py::arg("dtype") = "uint64");
        }
        py::object arr = cp.attr("empty")(n, py::arg("dtype") = "uint64");
        size_t* dst_ptr = get_device_pointer<size_t>(arr);
        CUDA_CHECK_THROW(cudaMemcpy(dst_ptr, candidates.query_candidate_offsets, n * sizeof(size_t),
                                    cudaMemcpyDeviceToDevice));
        return arr;
    }

    if (candidates.empty()) {
        return py::array_t<size_t>(0);
    }
    py::array_t<size_t> result(static_cast<py::ssize_t>(n));
    CUDA_CHECK_THROW(cudaMemcpy(result.mutable_data(), candidates.query_candidate_offsets,
                                n * sizeof(size_t), cudaMemcpyDeviceToHost));
    return result;
}

// --- Index getters ---
py::array_t<int> get_core_candidate_indices(const core::Index& index) {
    return copy_to_numpy(index.all_candidate_indices, index.n_total_candidates);
}

py::array_t<uint8_t> get_core_bucket_signatures(const core::Index& index) {
    size_t n = static_cast<size_t>(index.n_total_buckets) * index.sig_nbytes;
    return copy_to_numpy(index.all_bucket_signatures, n);
}

py::array_t<int> get_core_bucket_candidate_offsets(const core::Index& index) {
    return copy_to_numpy(index.bucket_candidate_offsets, index.n_total_buckets + 1);
}

py::array_t<int> get_core_table_bucket_offsets(const core::Index& index) {
    return copy_to_numpy(index.table_bucket_offsets, index.n_hash_tables + 1);
}

py::array_t<uint32_t> get_minhash_hash_a(const minhash::Index& index) {
    size_t n = static_cast<size_t>(index.core.n_hash_tables) * index.core.n_hashes;
    return copy_to_numpy(index.A, n);
}

py::array_t<uint32_t> get_minhash_hash_b(const minhash::Index& index) {
    size_t n = static_cast<size_t>(index.core.n_hash_tables) * index.core.n_hashes;
    return copy_to_numpy(index.B, n);
}

// --- Typed projection/bias getters (float/double dispatch) ---
template <typename IndexType> py::object get_projection_matrix(const IndexType& index) {
    size_t n =
        static_cast<size_t>(index.core.n_hash_tables) * index.core.n_hashes * index.core.n_features;
    return copy_to_numpy_typed(index.P, n, index.is_double);
}

py::object get_pslsh_bias(const pslsh::Index& index) {
    size_t n = static_cast<size_t>(index.core.n_hash_tables) * index.core.n_hashes;
    return copy_to_numpy_typed(index.b, n, index.is_double);
}

// --- Bind common core::Index properties/getters ---
template <typename IndexType, typename PyClass> void bind_core_index_properties(PyClass& cls) {
    cls.def("empty", &IndexType::empty)
        .def("size_bytes", &IndexType::size_bytes)
        .def_property_readonly("n_total_candidates",
                               [](const IndexType& idx) { return idx.core.n_total_candidates; })
        .def_property_readonly("n_total_buckets",
                               [](const IndexType& idx) { return idx.core.n_total_buckets; })
        .def_property_readonly("n_hash_tables",
                               [](const IndexType& idx) { return idx.core.n_hash_tables; })
        .def_property_readonly("n_hashes", [](const IndexType& idx) { return idx.core.n_hashes; })
        .def_property_readonly("sig_nbytes",
                               [](const IndexType& idx) { return idx.core.sig_nbytes; })
        .def_property_readonly("n_features",
                               [](const IndexType& idx) { return idx.core.n_features; })
        .def_property_readonly("seed", [](const IndexType& idx) { return idx.core.seed; })
        .def("get_candidate_indices",
             [](const IndexType& idx) { return get_core_candidate_indices(idx.core); })
        .def("get_bucket_signatures",
             [](const IndexType& idx) { return get_core_bucket_signatures(idx.core); })
        .def("get_bucket_candidate_offsets",
             [](const IndexType& idx) { return get_core_bucket_candidate_offsets(idx.core); })
        .def("get_table_bucket_offsets",
             [](const IndexType& idx) { return get_core_table_bucket_offsets(idx.core); });
}

// --- Index loaders ---
void load_core_index(core::Index& index, py::array_t<int>& candidate_indices,
                     py::array_t<uint8_t>& bucket_signatures,
                     py::array_t<int>& bucket_candidate_offsets,
                     py::array_t<int>& table_bucket_offsets, int n_total_candidates,
                     int n_total_buckets, int n_hash_tables, int n_hashes, int sig_nbytes,
                     int n_features, uint64_t seed) {
    // device arrays
    index.all_candidate_indices = make_device_array<int>(candidate_indices);
    index.all_bucket_signatures = make_device_array<uint8_t>(bucket_signatures);
    index.bucket_candidate_offsets = make_device_array<int>(bucket_candidate_offsets);
    index.table_bucket_offsets = make_device_array<int>(table_bucket_offsets);
    // metadata
    index.n_total_candidates = n_total_candidates;
    index.n_total_buckets = n_total_buckets;
    index.n_hash_tables = n_hash_tables;
    index.n_hashes = n_hashes;
    index.sig_nbytes = sig_nbytes;
    index.n_features = n_features;
    index.seed = seed;
}

template <typename T>
std::unique_ptr<rplsh::Index>
load_rplsh_index(py::array_t<int>& candidate_indices, py::array_t<uint8_t>& bucket_signatures,
                 py::array_t<int>& bucket_candidate_offsets, py::array_t<int>& table_bucket_offsets,
                 py::array_t<T>& projection, int n_total_candidates, int n_total_buckets,
                 int n_hash_tables, int n_hashes, int sig_nbytes, int n_features, uint64_t seed) {
    auto index = std::make_unique<rplsh::Index>();
    load_core_index(index->core, candidate_indices, bucket_signatures, bucket_candidate_offsets,
                    table_bucket_offsets, n_total_candidates, n_total_buckets, n_hash_tables,
                    n_hashes, sig_nbytes, n_features, seed);

    index->is_double = std::is_same_v<T, double>;
    index->P = make_device_array<T>(projection);
    return index;
}

std::unique_ptr<minhash::Index>
load_minhash_index(py::array_t<int> candidate_indices, py::array_t<uint8_t> bucket_signatures,
                   py::array_t<int> bucket_candidate_offsets, py::array_t<int> table_bucket_offsets,
                   py::array_t<uint32_t> hash_a, py::array_t<uint32_t> hash_b,
                   int n_total_candidates, int n_total_buckets, int n_hash_tables, int n_hashes,
                   int sig_nbytes, int n_features, uint64_t seed) {
    auto index = std::make_unique<minhash::Index>();
    load_core_index(index->core, candidate_indices, bucket_signatures, bucket_candidate_offsets,
                    table_bucket_offsets, n_total_candidates, n_total_buckets, n_hash_tables,
                    n_hashes, sig_nbytes, n_features, seed);

    index->A = make_device_array<uint32_t>(hash_a);
    index->B = make_device_array<uint32_t>(hash_b);
    return index;
}

template <typename T>
std::unique_ptr<pslsh::Index>
load_pslsh_index(py::array_t<int>& candidate_indices, py::array_t<uint8_t>& bucket_signatures,
                 py::array_t<int>& bucket_candidate_offsets, py::array_t<int>& table_bucket_offsets,
                 py::array_t<T>& projection, py::array_t<T>& bias, int n_total_candidates,
                 int n_total_buckets, int n_hash_tables, int n_hashes, int sig_nbytes,
                 int n_features, uint64_t seed) {
    auto index = std::make_unique<pslsh::Index>();
    load_core_index(index->core, candidate_indices, bucket_signatures, bucket_candidate_offsets,
                    table_bucket_offsets, n_total_candidates, n_total_buckets, n_hash_tables,
                    n_hashes, sig_nbytes, n_features, seed);

    index->is_double = std::is_same_v<T, double>;
    index->P = make_device_array<T>(projection);
    index->b = make_device_array<T>(bias);
    return index;
}

} // namespace python
} // namespace culsh

PYBIND11_MODULE(_culsh_core, m) {
    m.doc() = "Locality Sensitive Hashing on GPUs";

    using namespace culsh::python;

    // Candidates type
    py::class_<culsh::core::Candidates, std::unique_ptr<culsh::core::Candidates>>(m, "Candidates")
        .def("empty", &culsh::core::Candidates::empty)
        .def("get_indices", &get_candidate_indices, py::arg("as_cupy") = false)
        .def("get_counts", &get_candidate_counts, py::arg("as_cupy") = false)
        .def("get_offsets", &get_candidate_offsets, py::arg("as_cupy") = false)
        .def_readonly("n_queries", &culsh::core::Candidates::n_queries)
        .def_readonly("n_total_candidates", &culsh::core::Candidates::n_total_candidates);

    // RPLSH bindings
    auto rplsh_index =
        py::class_<culsh::rplsh::Index, std::unique_ptr<culsh::rplsh::Index>>(m, "RPLSHIndex");
    bind_core_index_properties<culsh::rplsh::Index>(rplsh_index);
    rplsh_index.def_readonly("is_double", &culsh::rplsh::Index::is_double)
        .def("get_projection_matrix", &get_projection_matrix<culsh::rplsh::Index>)
        .def_static("load_float", &load_rplsh_index<float>, py::arg("candidate_indices"),
                    py::arg("bucket_signatures"), py::arg("bucket_candidate_offsets"),
                    py::arg("table_bucket_offsets"), py::arg("projection"),
                    py::arg("n_total_candidates"), py::arg("n_total_buckets"),
                    py::arg("n_hash_tables"), py::arg("n_hashes"), py::arg("sig_nbytes"),
                    py::arg("n_features"), py::arg("seed"))
        .def_static("load_double", &load_rplsh_index<double>, py::arg("candidate_indices"),
                    py::arg("bucket_signatures"), py::arg("bucket_candidate_offsets"),
                    py::arg("table_bucket_offsets"), py::arg("projection"),
                    py::arg("n_total_candidates"), py::arg("n_total_buckets"),
                    py::arg("n_hash_tables"), py::arg("n_hashes"), py::arg("sig_nbytes"),
                    py::arg("n_features"), py::arg("seed"));

    py::class_<RPLSHCore>(m, "RPLSHCore")
        .def(py::init<int, int, uint64_t>(), py::arg("n_hash_tables"), py::arg("n_hashes"),
             py::arg("seed") = 42)
        .def("fit_float", &RPLSHCore::fit_float)
        .def("fit_double", &RPLSHCore::fit_double)
        .def("query_float", &RPLSHCore::query_float, py::arg("Q"), py::arg("n_queries"),
             py::arg("index"), py::arg("batch_size") = py::none())
        .def("query_double", &RPLSHCore::query_double, py::arg("Q"), py::arg("n_queries"),
             py::arg("index"), py::arg("batch_size") = py::none())
        .def("fit_query_float", &RPLSHCore::fit_query_float)
        .def("fit_query_double", &RPLSHCore::fit_query_double)
        .def_property_readonly("n_hash_tables", &RPLSHCore::n_hash_tables)
        .def_property_readonly("n_hashes", &RPLSHCore::n_hashes)
        .def_property_readonly("seed", &RPLSHCore::seed);

    // MinHash bindings
    auto minhash_index = py::class_<culsh::minhash::Index, std::unique_ptr<culsh::minhash::Index>>(
        m, "MinHashIndex");
    bind_core_index_properties<culsh::minhash::Index>(minhash_index);
    minhash_index.def("get_hash_a", &get_minhash_hash_a)
        .def("get_hash_b", &get_minhash_hash_b)
        .def_static("load", &load_minhash_index, py::arg("candidate_indices"),
                    py::arg("bucket_signatures"), py::arg("bucket_candidate_offsets"),
                    py::arg("table_bucket_offsets"), py::arg("hash_a"), py::arg("hash_b"),
                    py::arg("n_total_candidates"), py::arg("n_total_buckets"),
                    py::arg("n_hash_tables"), py::arg("n_hashes"), py::arg("sig_nbytes"),
                    py::arg("n_features"), py::arg("seed"));

    py::class_<MinHashCore>(m, "MinHashCore")
        .def(py::init<int, int, uint64_t>(), py::arg("n_hash_tables"), py::arg("n_hashes"),
             py::arg("seed") = 42)
        .def("fit", &MinHashCore::fit)
        .def("query", &MinHashCore::query, py::arg("indices"), py::arg("indptr"),
             py::arg("n_queries"), py::arg("index"))
        .def("fit_query", &MinHashCore::fit_query)
        .def_property_readonly("n_hash_tables", &MinHashCore::n_hash_tables)
        .def_property_readonly("n_hashes", &MinHashCore::n_hashes)
        .def_property_readonly("seed", &MinHashCore::seed);

    // PSLSH bindings
    auto pslsh_index =
        py::class_<culsh::pslsh::Index, std::unique_ptr<culsh::pslsh::Index>>(m, "PSLSHIndex");
    bind_core_index_properties<culsh::pslsh::Index>(pslsh_index);
    pslsh_index.def_readonly("is_double", &culsh::pslsh::Index::is_double)
        .def("get_projection_matrix", &get_projection_matrix<culsh::pslsh::Index>)
        .def("get_bias", &get_pslsh_bias)
        .def_static("load_float", &load_pslsh_index<float>, py::arg("candidate_indices"),
                    py::arg("bucket_signatures"), py::arg("bucket_candidate_offsets"),
                    py::arg("table_bucket_offsets"), py::arg("projection"), py::arg("bias"),
                    py::arg("n_total_candidates"), py::arg("n_total_buckets"),
                    py::arg("n_hash_tables"), py::arg("n_hashes"), py::arg("sig_nbytes"),
                    py::arg("n_features"), py::arg("seed"))
        .def_static("load_double", &load_pslsh_index<double>, py::arg("candidate_indices"),
                    py::arg("bucket_signatures"), py::arg("bucket_candidate_offsets"),
                    py::arg("table_bucket_offsets"), py::arg("projection"), py::arg("bias"),
                    py::arg("n_total_candidates"), py::arg("n_total_buckets"),
                    py::arg("n_hash_tables"), py::arg("n_hashes"), py::arg("sig_nbytes"),
                    py::arg("n_features"), py::arg("seed"));

    py::class_<PSLSHCore>(m, "PSLSHCore")
        .def(py::init<int, int, int, uint64_t>(), py::arg("n_hash_tables"), py::arg("n_hashes"),
             py::arg("window_size"), py::arg("seed") = 42)
        .def("fit_float", &PSLSHCore::fit_float)
        .def("fit_double", &PSLSHCore::fit_double)
        .def("query_float", &PSLSHCore::query_float, py::arg("Q"), py::arg("n_queries"),
             py::arg("index"), py::arg("batch_size") = py::none())
        .def("query_double", &PSLSHCore::query_double, py::arg("Q"), py::arg("n_queries"),
             py::arg("index"), py::arg("batch_size") = py::none())
        .def("fit_query_float", &PSLSHCore::fit_query_float)
        .def("fit_query_double", &PSLSHCore::fit_query_double)
        .def_property_readonly("n_hash_tables", &PSLSHCore::n_hash_tables)
        .def_property_readonly("n_hashes", &PSLSHCore::n_hashes)
        .def_property_readonly("window_size", &PSLSHCore::window_size)
        .def_property_readonly("seed", &PSLSHCore::seed);
}
