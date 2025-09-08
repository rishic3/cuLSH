# lsh

C++ LSH implementation. 

## Installation

```shell
# Download Eigen (https://eigen.tuxfamily.org/index.php?title=Main_Page)
./install_deps.sh

# Build project
./build.sh
```

## Usage

```cpp
#include <random_projection_lsh.h>
#include <Eigen/Dense>

int main() {
    RandomProjectionLSH lsh(16, 4);

    // read SIFT .fvecs format
    // see benchmark/bench_lsh.cpp for read_fvecs implementation
    Eigen::MatrixXd X = read_fvecs("../data/sift_base.fvecs");
    Eigen::MatrixXd Q = read_fvecs("../data/sift_query.fvecs");

    // fit - returns RandomProjectionLSHModel with index
    auto model = lsh.fit(X);

    // query - returns neighbor indices in X for each query
    auto all_neighbors = model.query_indices(Q);
}
```

## Benchmarking

Run the [download script](../download_sift1m.sh) to download the [SIFT1M dataset](http://corpus-texmex.irisa.fr/).  

```shell
# args: -d data_dir -h n_hash_tables -p n_projections -q n_queries -s seed
./_build/bench_lsh -d ../data/sift/ -h 64 -p 8 -q 10000 -s 42
```
