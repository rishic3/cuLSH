#!/bin/bash

set -e

BUILD_TYPE=${1:-Release}
BUILD_DIR="_build"
CXX=g++-12
CC=gcc-12
CUDA_HOST_COMPILER=g++-12

usage() {
    echo "Usage: $0 [Release|Debug] [--clean]"
    echo "  Release (default): Build with optimizations"
    echo "  Debug: Build with debug symbols"
    echo "  --clean: Remove build directory before building"
    exit 1
}

CLEAN=false
for arg in "$@"; do
    case $arg in
        --clean)
            CLEAN=true
            shift
            ;;
        Release|Debug)
            BUILD_TYPE=$arg
            ;;
        --help|-h)
            usage
            ;;
    esac
done

if [ "$CLEAN" = true ]; then
    echo "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "==================================================="
echo "Building Python bindings ($BUILD_TYPE)"
echo "==================================================="

cmake -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
      -DCMAKE_CXX_COMPILER=$CXX \
      -DCMAKE_C_COMPILER=$CC \
      -DCMAKE_CUDA_HOST_COMPILER=$CUDA_HOST_COMPILER \
      -DPython_EXECUTABLE=$(which python3) \
      -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=$(cd .. && pwd)/culsh \
      ..

make -j$(nproc)

echo ""
echo "Build complete! Extension installed to: culsh/_culsh_core*.so"
echo ""

