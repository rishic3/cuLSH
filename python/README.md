# lsh

Reference LSH implementation for Python. 

## Installation

```shell
pip install .
```

## Usage

```python
import numpy as np
from lsh.random_projection_lsh import RandomProjectionLSH, RandomProjectionLSHModel

def read_fvecs(fp):
    # read SIFT .fvecs format
    a = np.fromfile(fp, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy().view('float32')

X = read_fvecs('../data/sift/sift_base.fvecs')
Q = read_fvecs('../data/sift/sift_query.fvecs')

lsh = RandomProjectionLSH(n_hash_tables=16, n_projections=4)

# fit - returns RandomProjectionLSHModel with index
model = lsh.fit(X)

# query - returns neighbor indices in X for each query
all_neighbors = model.query(Q[:100])
```

## Benchmarking

Run the [download script](../download_sift1m.sh) to download the [SIFT1M dataset](http://corpus-texmex.irisa.fr/).

```shell
./run_benchmark -d ../data/sift --n-hash-tables 64 --n-projections 8 --n-queries 10000
```
