# CUDA Implementation

Core CUDA implementation of LSH algorithms.

## Building

```bash
# Builds _build_debug or _build_release
make [debug|release]
```

## Design

```
cuda/
├── culsh/
│   ├── include/culsh/       # Public headers
│   │   ├── minhash/           # MinHash LSH API
│   │   └── rplsh/             # Random Projection LSH API
│   └── src/                 # Implementation
│       ├── core/              # Shared kernels & utilities
│       ├── minhash/           # MinHash kernels
│       └── rplsh/             # Random Projection kernels
└── benchmark/               # CUDA benchmarks
```

LSH algorithms differ primarily in how they compute the signature matrix `X_sig`. Once signatures are computed, fitting (building hash tables) and querying (finding candidates) are shared across algorithms. The `core/` directory contains these common kernels, while each algorithm (minhash, rplsh) implements its own hashing kernels.
