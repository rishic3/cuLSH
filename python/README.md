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
    # read .fvecs from SIFT: http://corpus-texmex.irisa.fr/
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

```shell
# Download sift1m to data/sift
python benchmark/download_sift1m.py

# Run benchmark
./run_benchmark -d data/sift --n-hash-tables 64 --n-projections 8 --n-queries 10000
```
