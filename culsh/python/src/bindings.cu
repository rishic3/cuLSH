#include "common/array_utils.cuh"
#include "common/cuda_utils.cuh"

#include <culsh/rplsh/params.hpp>
#include <culsh/rplsh/rplsh.hpp>
#include <rplsh/candidates.cuh>
#include <rplsh/index.cuh>

namespace py = pybind11;

namespace culsh {
namespace python {

class RPLSHCore : public CUDAResourceManager {
  private:
    int n_hash_tables_ = 0;
    int n_projections_ = 0;
    uint64_t seed_ = 0;

  public:
    RPLSHCore(int n_hash_tables, int n_projections, uint64_t seed = 42)
        : CUDAResourceManager(),
          n_hash_tables_(n_hash_tables),
          n_projections_(n_projections),
          seed_(seed) {}

    std::unique_ptr<rplsh::Index> fit_float(py::object X_obj, int n_samples, int n_features) {
        float* X_ptr = get_device_pointer<float>(X_obj);

        culsh::RPLSHParams params{n_hash_tables_, n_projections_, seed_};
        auto index = rplsh::fit(cublas_handle_, stream_, X_ptr, n_samples, n_features, params);

        synchronize();
        return std::make_unique<rplsh::Index>(std::move(index));
    }

    std::unique_ptr<rplsh::Index> fit_double(py::object X_obj, int n_samples, int n_features) {
        double* X_ptr = get_device_pointer<double>(X_obj);

        culsh::RPLSHParams params{n_hash_tables_, n_projections_, seed_};
        auto index = rplsh::fit(cublas_handle_, stream_, X_ptr, n_samples, n_features, params);

        synchronize();
        return std::make_unique<rplsh::Index>(std::move(index));
    }

    std::unique_ptr<rplsh::Candidates> query_float(py::object Q_obj, int n_queries,
                                                   const rplsh::Index& index) {
        if (index.is_double) {
            throw std::runtime_error("Index was fitted with float64, but query is float32");
        }

        float* Q_ptr = get_device_pointer<float>(Q_obj);
        auto candidates = rplsh::query_indices(cublas_handle_, stream_, Q_ptr, n_queries, index);

        synchronize();
        return std::make_unique<rplsh::Candidates>(std::move(candidates));
    }

    std::unique_ptr<rplsh::Candidates> query_double(py::object Q_obj, int n_queries,
                                                    const rplsh::Index& index) {
        if (!index.is_double) {
            throw std::runtime_error("Index was fitted with float32, but query is float64");
        }

        double* Q_ptr = get_device_pointer<double>(Q_obj);
        auto candidates = rplsh::query_indices(cublas_handle_, stream_, Q_ptr, n_queries, index);

        synchronize();
        return std::make_unique<rplsh::Candidates>(std::move(candidates));
    }

    int n_hash_tables() const { return n_hash_tables_; }
    int n_projections() const { return n_projections_; }
    uint64_t seed() const { return seed_; }
};

py::array_t<int> get_candidate_indices(const rplsh::Candidates& candidates) {
    if (candidates.empty()) {
        return py::array_t<int>(0);
    }
    py::array_t<int> result(static_cast<py::ssize_t>(candidates.n_total_candidates));
    CUDA_CHECK_THROW(cudaMemcpy(result.mutable_data(), candidates.query_candidate_indices,
                                candidates.n_total_candidates * sizeof(int),
                                cudaMemcpyDeviceToHost));
    return result;
}

py::array_t<size_t> get_candidate_counts(const rplsh::Candidates& candidates) {
    if (candidates.empty()) {
        return py::array_t<size_t>(0);
    }
    py::array_t<size_t> result(static_cast<py::ssize_t>(candidates.n_queries));
    CUDA_CHECK_THROW(cudaMemcpy(result.mutable_data(), candidates.query_candidate_counts,
                                candidates.n_queries * sizeof(size_t), cudaMemcpyDeviceToHost));
    return result;
}

py::array_t<size_t> get_candidate_offsets(const rplsh::Candidates& candidates) {
    if (candidates.empty()) {
        return py::array_t<size_t>(0);
    }
    py::array_t<size_t> result(static_cast<py::ssize_t>(candidates.n_queries + 1));
    CUDA_CHECK_THROW(cudaMemcpy(result.mutable_data(), candidates.query_candidate_offsets,
                                (candidates.n_queries + 1) * sizeof(size_t),
                                cudaMemcpyDeviceToHost));
    return result;
}

} // namespace python
} // namespace culsh


PYBIND11_MODULE(_culsh_core, m) {
    m.doc() = "Locality Sensitive Hashing on GPUs";

    using namespace culsh::python;
    using namespace culsh::rplsh;

    py::class_<Index, std::unique_ptr<Index>>(m, "Index")
        .def("empty", &Index::empty)
        .def("device_size", &Index::device_size)
        .def_readonly("n_total_candidates", &Index::n_total_candidates)
        .def_readonly("n_total_buckets", &Index::n_total_buckets)
        .def_readonly("n_hash_tables", &Index::n_hash_tables)
        .def_readonly("n_projections", &Index::n_projections)
        .def_readonly("n_features", &Index::n_features)
        .def_readonly("seed", &Index::seed)
        .def_readonly("is_double", &Index::is_double);

    py::class_<Candidates, std::unique_ptr<Candidates>>(m, "Candidates")
        .def("empty", &Candidates::empty)
        .def("get_indices", &get_candidate_indices)
        .def("get_counts", &get_candidate_counts)
        .def("get_offsets", &get_candidate_offsets)
        .def_readonly("n_queries", &Candidates::n_queries)
        .def_readonly("n_total_candidates", &Candidates::n_total_candidates);

    py::class_<RPLSHCore>(m, "RPLSHCore")
        .def(py::init<int, int, uint64_t>(),
             py::arg("n_hash_tables"), py::arg("n_projections"), py::arg("seed") = 42)
        .def("fit_float", &RPLSHCore::fit_float)
        .def("fit_double", &RPLSHCore::fit_double)
        .def("query_float", &RPLSHCore::query_float)
        .def("query_double", &RPLSHCore::query_double)
        .def_property_readonly("n_hash_tables", &RPLSHCore::n_hash_tables)
        .def_property_readonly("n_projections", &RPLSHCore::n_projections)
        .def_property_readonly("seed", &RPLSHCore::seed);
}
