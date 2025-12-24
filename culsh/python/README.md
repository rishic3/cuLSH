# Python Bindings

The Python bindings provide a Python API over the CUDA LSH implementations using [pybind11](https://pybind11.readthedocs.io/en/stable/).  

## Installation

```bash
pip install .
```

## Development

```bash
pip install -e .

# If making C++ changes, rebuild .so
make clean
make [debug|release]
```

## Design

The Python layer accepts numpy or cupy arrays (numpy inputs are copied immediately to cupy). These are passed as device pointers to the CUDA layer via the [`__cuda_array_interface__`](https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html).

The CUDA layer produces containers holding pointers to device memory: `Index` and `Candidates` for fit and query respectively. To manage memory, the CUDA layer implements RAII semantics over these containers, and the Python bindings wrap them in a unique ptr.

The output candidates returns a CSR-like result (indices, offsets) pointing to samples in `X`, the fitted array.

```
Python (culsh/)          C++ Bindings (src/)           CUDA (../cuda/)
─────────────────        ───────────────────           ───────────────
RPLSH, RPLSHModel   →    RPLSHCore (pybind11)     →    rplsh::fit/query
     ↓                        ↓                             ↓
numpy/cupy arrays        __cuda_array_interface__      Index, Candidates
```
