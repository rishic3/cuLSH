#!/bin/bash

set -e

BUILD_DIR="_build_profile"

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCMAKE_CXX_FLAGS="-pg -g -O2" \
      -DCMAKE_EXE_LINKER_FLAGS="-pg" \
      ..

make -j$(nproc)

cd ..

echo "Built bench_lsh: $BUILD_DIR/bench_lsh"

if [ $# -eq 0 ]; then
    echo "Please provide bench_lsh args."
    exit 1
fi

echo "Running bench_lsh..."
echo "==================================================="

if ./$BUILD_DIR/bench_lsh "$@"; then
    echo "==================================================="
    echo "Generating gprof profile..."
    
    mkdir -p profile
    DT=$(date '+%Y%m%d_%H%M%S')
    profile_path=profile/profile_report_${DT}.txt
    
    gprof ./$BUILD_DIR/bench_lsh gmon.out > $profile_path
    echo "gprof profile saved to: $(pwd)/$profile_path"
else
    echo "==================================================="
    echo "Benchmark failed"
    exit 1
fi
