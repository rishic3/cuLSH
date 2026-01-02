#pragma once

#include "cuda_utils.cuh"

#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace culsh {
namespace python {

/**
 * @brief Get device pointer from a cupy array
 */
template <typename T>
T* get_device_pointer(py::object obj) {
    if (!py::hasattr(obj, "__cuda_array_interface__")) {
        throw std::runtime_error("Expected cupy array with __cuda_array_interface__.");
    }

    py::dict interface = obj.attr("__cuda_array_interface__").cast<py::dict>();
    // returns (int, bool) of (data pointer, read-only)
    py::tuple data = interface["data"].cast<py::tuple>();
    uintptr_t ptr = data[0].cast<uintptr_t>();

    return reinterpret_cast<T*>(ptr);
}

/**
 * @brief Copy numpy array to device memory and return pointer
 */
template <typename T>
T* make_device_array(const py::array_t<T>& arr) {
    size_t n = static_cast<size_t>(arr.size());
    if (n == 0) return nullptr;

    T* ptr = nullptr;
    CUDA_CHECK_THROW(cudaMalloc(&ptr, n * sizeof(T)));
    CUDA_CHECK_THROW(cudaMemcpy(ptr, arr.data(), n * sizeof(T), cudaMemcpyHostToDevice));
    return ptr;
}

/**
 * @brief Copy device array to numpy array
 */
template <typename T>
py::array_t<T> copy_to_numpy(const T* device_ptr, size_t n) {
    if (device_ptr == nullptr || n == 0) {
        return py::array_t<T>(0);
    }
    py::array_t<T> result(static_cast<py::ssize_t>(n));
    CUDA_CHECK_THROW(cudaMemcpy(result.mutable_data(), device_ptr, n * sizeof(T), cudaMemcpyDeviceToHost));
    return result;
}

/**
 * @brief Copy device array to numpy with float/double dispatch
 */
inline py::object copy_to_numpy_typed(const void* device_ptr, size_t n, bool is_double) {
    if (is_double) {
        return copy_to_numpy(static_cast<const double*>(device_ptr), n);
    } else {
        return copy_to_numpy(static_cast<const float*>(device_ptr), n);
    }
}

} // namespace python
} // namespace culsh
