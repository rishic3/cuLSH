cmake_minimum_required(VERSION 3.12)
project(cuLSH VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
set(EIGEN_DIR "${THIRD_PARTY_DIR}/eigen")

if(NOT EXISTS "${EIGEN_DIR}")
    message(WARNING "Eigen not found at ${EIGEN_DIR}. Run 'make install-deps' first.")
endif()

include_directories(${EIGEN_DIR})

include_directories(include)

add_library(culsh_lib 
    src/random_projection_lsh.cpp
)

# main
add_executable(main src/main.cpp)
target_link_libraries(main culsh_lib)

# benchmarks
add_executable(bench_lsh benchmark/bench_lsh.cpp)
target_link_libraries(bench_lsh culsh_lib)

# build type
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# default to
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Custom targets for convenience (like your Makefile)
add_custom_target(install-deps
    COMMAND mkdir -p ${THIRD_PARTY_DIR}
    COMMAND cd ${THIRD_PARTY_DIR} && 
            wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz &&
            tar -xzf eigen-3.4.0.tar.gz &&
            mv eigen-3.4.0 eigen
    COMMENT "Downloading and installing Eigen 3.4.0..."
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_custom_target(run-main
    COMMAND main
    DEPENDS main
    COMMENT "Running main executable..."
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_target(run-bench
    COMMAND bench_lsh
    DEPENDS bench_lsh
    COMMENT "Running benchmark..."
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
