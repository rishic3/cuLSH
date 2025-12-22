# Python Bindings

The Python class is a thin wrapper over the CUDA LSH fit and query methods. It accepts numpy or cupy arrays  
(the former of which is copied to cupy).  
We use pybind11 to interface with LSH methods. The device pointers for inputs are passed directly via the pybind11 `__cuda_array_interface__`.  

LSH fit produces containers that hold device memory - `Index` and `Candidates` for fit and query respectively. To manage memory, the CUDA layer  
implements RAII semantics over these containers, and the Python bindings wrap them in a unique ptr.

```
Python (culsh/)          C++ Bindings (src/)           CUDA (../cuda/)
─────────────────        ───────────────────           ───────────────
RPLSH, RPLSHModel   →    RPLSHCore (pybind11)     →    rplsh::fit/query
     ↓                        ↓                             ↓
numpy/cupy arrays        __cuda_array_interface__      Index, Candidates
```
