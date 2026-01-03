# cuLSH

**GPU-accelerated Locality Sensitive Hashing**

cuLSH is a library for ANN-search using LSH.

## Supported Algorithms

| Algorithm | Class | Distance Metric | Data Type |
|-----------|-------|-----------------|-----------|
| p-Stable LSH | [`PStableLSH`](api/pstable.md) | Euclidean | Dense |
| Random Projection LSH | [`RPLSH`](api/rplsh.md) | Cosine | Dense |
| MinHash LSH | [`MinHashLSH`](api/minhash.md) | Jaccard | Sparse |

## Quick Example

```python
import numpy as np
from culsh import PStableLSH

# Create data
X = np.random.randn(10000, 128).astype(np.float32)
Q = np.random.randn(100, 128).astype(np.float32)

# Fit and query
model = PStableLSH(n_hash_tables=16, n_hashes=8).fit(X)
candidates = model.query(Q)

# Access results
indices = candidates.get_indices()
counts = candidates.get_counts()
```
