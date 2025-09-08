#!/bin/bash

set -e

BUILD_TYPE=${1:-Release}
BUILD_DIR="_build"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake -DCMAKE_BUILD_TYPE="$BUILD_TYPE" ..
make -j$(nproc)

echo "Built main: $BUILD_DIR/main"
echo "Built bench_lsh: $BUILD_DIR/bench_lsh"
