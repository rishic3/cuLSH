#pragma once

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
    py::tuple data = interface["data"].cast<py::tuple>();
    uintptr_t ptr = data[0].cast<uintptr_t>();

    return reinterpret_cast<T*>(ptr);
}

} // namespace python
} // namespace culsh
