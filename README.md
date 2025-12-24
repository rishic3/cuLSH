# cuLSH [WIP]

Locality Sensitive Hashing on GPUs with Python bindings.

## Getting Started

### Installation

```bash
cd culsh/python
pip install .
```

### Development

```bash
cd culsh/python
pip install -e .

# If making C++ changes, rebuild .so
make clean
make [debug|release]
```

## Usage

### Fit and Query (NumPy)

```python
import numpy as np
from culsh import RPLSH

X = np.random.randn(100, 128).astype(np.float32)
Q = np.random.randn(10, 128).astype(np.float32)

# Fit (returns RPLSHModel)
model = RPLSH(n_hash_tables=16, n_projections=8, seed=42).fit(X)

# Query (returns candidate neighbors)
candidates = model.query(Q)

# Get neighbors
indices = candidates.get_indices()  # [15, 73, 14, 29, 35, ...]
offsets = candidates.get_offsets()  # [0, 2, 9, 14, 21, ...]
counts = candidates.get_counts()  # [2, 7, 5, 7, 8, ...]
```

### Fit and Query (CuPy)

```python
import cupy as cp
from culsh import RPLSH

X = cp.random.randn(100, 128).astype(cp.float32)
Q = cp.random.randn(10, 128).astype(cp.float32)

# Fit (returns RPLSHModel)
model = RPLSH(n_hash_tables=16, n_projections=8, seed=42).fit(X)

# Query (returns candidate neighbors)
candidates = model.query(Q)

# Get neighbors
indices = candidates.get_indices(as_cupy=True)
offsets = candidates.get_offsets(as_cupy=True)
counts = candidates.get_counts(as_cupy=True)
```

### Simultaneous Fit + Query

```python
import numpy as np
from culsh import RPLSH

X = np.random.randn(100, 128).astype(np.float32)

# Fit + Query
candidates = RPLSH(n_hash_tables=16, n_projections=8, seed=42).fit_query(X)

# Get neighbors
indices = candidates.get_indices()
offsets = candidates.get_offsets()
counts = candidates.get_counts()
```

### Batched Queries

For large query sets, use `batch_size` to reduce peak GPU memory:

```python
# Process 100 queries at a time, returns single merged Candidates
candidates = model.query(Q, batch_size=100)
```
