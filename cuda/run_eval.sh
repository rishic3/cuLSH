#!/bin/bash

set -e

BUILD_DIR="_build"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)

cd ..

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
HASH_TABLES=(16 32 64 128)
PROJECTIONS=(4 4 8 8)

echo "Running benchmark with ${#HASH_TABLES[@]} parameter combinations..."

for i in "${!HASH_TABLES[@]}"; do
    h=${HASH_TABLES[$i]}
    p=${PROJECTIONS[$i]}
    echo "Running benchmark with h=${h}, p=${p}..."
    ./_build/benchmark -h $h -p $p -s 42 -o eval/eval_${TIMESTAMP}
done

echo "All benchmarks completed!"
