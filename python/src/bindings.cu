#include "common/array_utils.cuh"
#include "common/cuda_utils.cuh"

#include <culsh/minhash/minhash.hpp>
#include <culsh/minhash/params.hpp>
#include <culsh/rplsh/params.hpp>
#include <culsh/rplsh/rplsh.hpp>
#include <core/candidates.cuh>
#include <core/index.cuh>
#include <minhash/index.cuh>
#include <rplsh/index.cuh>

#include <optional>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace culsh {
namespace python {

class RPLSHCore : public CUDAResourceManager {
  private:
    int n_hash_tables_ = 0;
    int n_hashes_ = 0;
    uint64_t seed_ = 0;

  public:
    RPLSHCore(int n_hash_tables, int n_hashes, uint64_t seed = 42)
        : CUDAResourceManager(),
          n_hash_tables_(n_hash_tables),
          n_hashes_(n_hashes),
          seed_(seed) {}

    std::unique_ptr<rplsh::Index> fit_float(py::object X_obj, int n_samples, int n_features) {
        float* X_ptr = get_device_pointer<float>(X_obj);

        culsh::rplsh::RPLSHParams params{n_hash_tables_, n_hashes_, seed_};
        auto index = rplsh::fit(cublas_handle_, stream_, X_ptr, n_samples, n_features, params);

        synchronize();
        return std::make_unique<rplsh::Index>(std::move(index));
    }

    std::unique_ptr<rplsh::Index> fit_double(py::object X_obj, int n_samples, int n_features) {
        double* X_ptr = get_device_pointer<double>(X_obj);

        culsh::rplsh::RPLSHParams params{n_hash_tables_, n_hashes_, seed_};
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
            ? rplsh::query_batched(cublas_handle_, stream_, Q_ptr, n_queries, index, batch_size.value())
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
            ? rplsh::query_batched(cublas_handle_, stream_, Q_ptr, n_queries, index, batch_size.value())
            : rplsh::query(cublas_handle_, stream_, Q_ptr, n_queries, index);

        synchronize();
        return std::make_unique<rplsh::Candidates>(std::move(candidates));
    }

    std::unique_ptr<rplsh::Candidates> fit_query_float(py::object X_obj, int n_samples,
                                                       int n_features) {
        float* X_ptr = get_device_pointer<float>(X_obj);

        culsh::rplsh::RPLSHParams params{n_hash_tables_, n_hashes_, seed_};
        auto candidates = rplsh::fit_query(cublas_handle_, stream_, X_ptr, n_samples, n_features, params);

        synchronize();
        return std::make_unique<rplsh::Candidates>(std::move(candidates));
    }

    std::unique_ptr<rplsh::Candidates> fit_query_double(py::object X_obj, int n_samples,
                                                        int n_features) {
        double* X_ptr = get_device_pointer<double>(X_obj);

        culsh::rplsh::RPLSHParams params{n_hash_tables_, n_hashes_, seed_};
        auto candidates = rplsh::fit_query(cublas_handle_, stream_, X_ptr, n_samples, n_features, params);

        synchronize();
        return std::make_unique<rplsh::Candidates>(std::move(candidates));
    }

    int n_hash_tables() const { return n_hash_tables_; }
    int n_hashes() const { return n_hashes_; }
    uint64_t seed() const { return seed_; }
};

class MinHashCore : public CUDAResourceManager {
  private:
    int n_hash_tables_ = 0;
    int n_hashes_ = 0;
    uint64_t seed_ = 0;

  public:
    MinHashCore(int n_hash_tables, int n_hashes, uint64_t seed = 42)
        : CUDAResourceManager(),
          n_hash_tables_(n_hash_tables),
          n_hashes_(n_hashes),
          seed_(seed) {}

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
        auto candidates = minhash::fit_query(stream_, indices_ptr, indptr_ptr, n_samples, n_features, params);

        synchronize();
        return std::make_unique<minhash::Candidates>(std::move(candidates));
    }

    int n_hash_tables() const { return n_hash_tables_; }
    int n_hashes() const { return n_hashes_; }
    uint64_t seed() const { return seed_; }
};

py::object get_candidate_indices(const core::Candidates& candidates, bool as_cupy = false) {
    size_t n = candidates.n_total_candidates;

    if (as_cupy) {
        py::module_ cp = py::module_::import("cupy");
        if (candidates.empty()) {
            return cp.attr("empty")(0, py::arg("dtype") = "int32");
        }
        py::object arr = cp.attr("empty")(n, py::arg("dtype") = "int32");
        int* dst_ptr = get_device_pointer<int>(arr);
        CUDA_CHECK_THROW(cudaMemcpy(dst_ptr, candidates.query_candidate_indices,
                                    n * sizeof(int), cudaMemcpyDeviceToDevice));
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
        CUDA_CHECK_THROW(cudaMemcpy(dst_ptr, candidates.query_candidate_counts,
                                    n * sizeof(size_t), cudaMemcpyDeviceToDevice));
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
        CUDA_CHECK_THROW(cudaMemcpy(dst_ptr, candidates.query_candidate_offsets,
                                    n * sizeof(size_t), cudaMemcpyDeviceToDevice));
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
    py::class_<culsh::rplsh::Index, std::unique_ptr<culsh::rplsh::Index>>(m, "RPLSHIndex")
        .def("empty", &culsh::rplsh::Index::empty)
        .def("size_bytes", &culsh::rplsh::Index::size_bytes)
        .def_property_readonly("n_total_candidates", [](const culsh::rplsh::Index& idx) { return idx.core.n_total_candidates; })
        .def_property_readonly("n_total_buckets", [](const culsh::rplsh::Index& idx) { return idx.core.n_total_buckets; })
        .def_property_readonly("n_hash_tables", [](const culsh::rplsh::Index& idx) { return idx.core.n_hash_tables; })
        .def_property_readonly("n_hashes", [](const culsh::rplsh::Index& idx) { return idx.core.n_hashes; })
        .def_property_readonly("sig_nbytes", [](const culsh::rplsh::Index& idx) { return idx.core.sig_nbytes; })
        .def_property_readonly("n_features", [](const culsh::rplsh::Index& idx) { return idx.core.n_features; })
        .def_property_readonly("seed", [](const culsh::rplsh::Index& idx) { return idx.core.seed; })
        .def_readonly("is_double", &culsh::rplsh::Index::is_double);

    py::class_<RPLSHCore>(m, "RPLSHCore")
        .def(py::init<int, int, uint64_t>(),
             py::arg("n_hash_tables"), py::arg("n_hashes"), py::arg("seed") = 42)
        .def("fit_float", &RPLSHCore::fit_float)
        .def("fit_double", &RPLSHCore::fit_double)
        .def("query_float", &RPLSHCore::query_float,
             py::arg("Q"), py::arg("n_queries"), py::arg("index"), py::arg("batch_size") = py::none())
        .def("query_double", &RPLSHCore::query_double,
             py::arg("Q"), py::arg("n_queries"), py::arg("index"), py::arg("batch_size") = py::none())
        .def("fit_query_float", &RPLSHCore::fit_query_float)
        .def("fit_query_double", &RPLSHCore::fit_query_double)
        .def_property_readonly("n_hash_tables", &RPLSHCore::n_hash_tables)
        .def_property_readonly("n_hashes", &RPLSHCore::n_hashes)
        .def_property_readonly("seed", &RPLSHCore::seed);

    // MinHash bindings
    py::class_<culsh::minhash::Index, std::unique_ptr<culsh::minhash::Index>>(m, "MinHashIndex")
        .def("empty", &culsh::minhash::Index::empty)
        .def("size_bytes", &culsh::minhash::Index::size_bytes)
        .def_property_readonly("n_total_candidates", [](const culsh::minhash::Index& idx) { return idx.core.n_total_candidates; })
        .def_property_readonly("n_total_buckets", [](const culsh::minhash::Index& idx) { return idx.core.n_total_buckets; })
        .def_property_readonly("n_hash_tables", [](const culsh::minhash::Index& idx) { return idx.core.n_hash_tables; })
        .def_property_readonly("n_hashes", [](const culsh::minhash::Index& idx) { return idx.core.n_hashes; })
        .def_property_readonly("sig_nbytes", [](const culsh::minhash::Index& idx) { return idx.core.sig_nbytes; })
        .def_property_readonly("n_features", [](const culsh::minhash::Index& idx) { return idx.core.n_features; })
        .def_property_readonly("seed", [](const culsh::minhash::Index& idx) { return idx.core.seed; });

    py::class_<MinHashCore>(m, "MinHashCore")
        .def(py::init<int, int, uint64_t>(),
             py::arg("n_hash_tables"), py::arg("n_hashes"), py::arg("seed") = 42)
        .def("fit", &MinHashCore::fit)
        .def("query", &MinHashCore::query,
             py::arg("indices"), py::arg("indptr"), py::arg("n_queries"), py::arg("index"))
        .def("fit_query", &MinHashCore::fit_query)
        .def_property_readonly("n_hash_tables", &MinHashCore::n_hash_tables)
        .def_property_readonly("n_hashes", &MinHashCore::n_hashes)
        .def_property_readonly("seed", &MinHashCore::seed);
}
