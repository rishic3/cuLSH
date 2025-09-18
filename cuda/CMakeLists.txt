cmake_minimum_required(VERSION 3.18)
project(cuLSH VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_library(culsh_lib 
    culsh/src/rplsh/rplsh.cu
)

# include directories
target_include_directories(culsh_lib PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/culsh/include
)

# cuda required libraries
target_link_libraries(culsh_lib
    CUDA::curand
    CUDA::cudart
    CUDA::cublas
)

set_target_properties(culsh_lib PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# main
add_executable(main culsh/src/main.cu)
target_link_libraries(main culsh_lib)

# build type
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0")

set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86")

# default to
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
