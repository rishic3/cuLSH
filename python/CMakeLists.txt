cmake_minimum_required(VERSION 3.18)

set(CMAKE_C_COMPILER "/usr/bin/gcc-12")
set(CMAKE_CXX_COMPILER "/usr/bin/g++-12")
set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++-12")

project(culsh_python LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG QUIET)

# If pybind11 not found via config, try to find it via Python
if(NOT pybind11_FOUND)
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE pybind11_RESULT
    )
    if(pybind11_RESULT EQUAL 0)
        find_package(pybind11 CONFIG REQUIRED PATHS ${pybind11_DIR})
    else()
        message(FATAL_ERROR "pybind11 not found")
    endif()
endif()

# CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

# Path to CUDA library source
set(CUDA_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../cuda")

# Build the pybind11 module
pybind11_add_module(_culsh_core
    csrc/bindings.cu
    ${CUDA_LIB_DIR}/culsh/src/rplsh/rplsh.cu
    ${CUDA_LIB_DIR}/culsh/src/minhash/minhash.cu
    ${CUDA_LIB_DIR}/culsh/src/pslsh/pslsh.cu
)

# Include directories
target_include_directories(_culsh_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUDA_LIB_DIR}/culsh/include
    ${CUDA_LIB_DIR}/culsh/src
)

# Link CUDA libraries
target_link_libraries(_culsh_core PRIVATE
    CUDA::curand
    CUDA::cudart
    CUDA::cublas
)

# Enable position independent code for shared library
set_target_properties(_culsh_core PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
)

# Compiler flags
target_compile_options(_culsh_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr>
)
