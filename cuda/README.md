# CUDA Implementation

Core CUDA implementation of LSH algorithms.

## Building

```bash
# Builds _build_debug or _build_release
make [debug|release]
```

## Design

Computationally, LSH algorithms differ only in how they compute the signature matrix `X_sig`/`Q_sig`. Each algorithm implements its own hashing kernel(s), expected to produce the signature matrix in 'table-major' format:

```text
X_sig: (n_hash_tables × n_samples × n_hashes)

[ ---------- Table 0 ---------- | ---------- Table 1 ---------- | ... ]

┌──────────────────────────────────────────────────────────────┐
│  Table 0                      │  Table 1                     │
│  ┌─────────────────────────┐  │  ┌─────────────────────────┐ │
│  │ sample 0: [h0 h1 h2 h3] │  │  │ sample 0: [h0 h1 h2 h3] │ │
│  │ sample 1: [h0 h1 h2 h3] │  │  │ sample 1: [h0 h1 h2 h3] │ │
│  │ sample 2: [h0 h1 h2 h3] │  │  │ sample 2: [h0 h1 h2 h3] │ │
│  │ ...                     │  │  │ ...                     │ │
│  └─────────────────────────┘  │  └─────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

Once `X_sig`/`Q_sig` is computed by the algorithm-specific hashing kernel(s), building the index and querying the index is identical across algorithms. The `core/` directory contains these common kernels. 

Algorithms however use different signature dtypes — e.g. Random Projection LSH uses binary `uint8_t` signatures, PStableLSH uses signed `int32_t` signatures, etc. The core fit/query kernels accept any fixed-width type, treating `X_sig`/`Q_sig` as opaque bytes, since byte-level equality is the only concern for signature matching. Only the width of a signature — i.e., `sizeof(hash_bit) * n_hashes` — must be specified.

Algorithms must also implement a thin Index wrapper. This owns the `core::Index`, containing the actual index on device, as well as the algorithm-specific hash functions used during fit (e.g. projection vectors). The latter are used to hash the query data at query-time. The wrapper (and the core index) implement RAII semantics to free device memory on destruction. This way, the Python bindings simply wrap the struct in a unique ptr for memory management.

```
cuda/
└── culsh/
    ├── include/culsh/       # Public headers
    │   ├── minhash/           # MinHash LSH API
    │   ├── rplsh/             # Random Projection LSH API
    │   └── .../
    └── src/                 # Implementation
        ├── core/              # Shared kernels & utilities
        ├── minhash/           # MinHash kernels
        ├── rplsh/             # Random Projection kernels
        └── .../
```
