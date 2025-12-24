#!/bin/bash

set -e

BUILD_TYPE=${1:-Release}
BUILD_DIR="_build"

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# cmake -DCMAKE_BUILD_TYPE="$BUILD_TYPE" ..
cmake -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
      -DCMAKE_CXX_COMPILER=g++-12 \
      -DCMAKE_C_COMPILER=gcc-12 \
      -DCMAKE_CUDA_HOST_COMPILER=g++-12 \
      ..
make -j$(nproc)

echo "Built main: $BUILD_DIR/main"
