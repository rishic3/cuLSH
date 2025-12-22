# cuLSH [WIP]

Locality Sensitive Hashing on GPUs with Python bindings.

## Installation

```bash
cd culsh/python
pip install .
```

## Development

```bash
pip install -e .

# If making C++ changes, rebuild .so
make clean
make [debug|release]
```

## Usage

```python
import numpy as np
from culsh import RPLSH

# Fit
X = np.random.randn(10000, 128).astype(np.float32)
model = RPLSH(n_hash_tables=16, n_projections=8).fit(X)

# Query
Q = np.random.randn(100, 128).astype(np.float32)
candidates = model.query(Q)

# Get candidates for each query
indices = candidates.get_indices()
offsets = candidates.get_offsets()
for i in range(len(Q)):
    query_candidates = indices[offsets[i]:offsets[i + 1]]
```
