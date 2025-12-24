#!/bin/bash

set -e

BUILD_TYPE=${1:-Release}
BUILD_DIR="_build"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake -DCMAKE_BUILD_TYPE="$BUILD_TYPE" ..
make -j$(nproc)

echo

./main
