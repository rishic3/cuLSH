from typing import Any, Callable, Union

import cupy as cp
import cupyx.scipy.sparse
import numpy as np
import scipy.sparse

from culsh.utils import compute_recall

ArrayLike = Union[
    np.ndarray, cp.ndarray, scipy.sparse.csr_matrix, cupyx.scipy.sparse.csr_matrix
]


def generate_dense_data(
    n_samples: int,
    n_features: int,
    dtype: Any = np.float32,
    seed: int = 42,
    cupy: bool = False,
) -> Union[np.ndarray, cp.ndarray]:
    """Generate random dense matrix"""
    if cupy:
        cp.random.seed(seed)
        return cp.random.randn(n_samples, n_features).astype(dtype)
    else:
        np.random.seed(seed)
        return np.random.randn(n_samples, n_features).astype(dtype)


def generate_sparse_data(
    n_samples: int,
    n_features: int,
    density: float,
    dtype: Any = np.float32,
    seed: int = 42,
    cupy: bool = False,
) -> Union[scipy.sparse.csr_matrix, cupyx.scipy.sparse.csr_matrix]:
    """Generate random sparse binary CSR matrix"""
    if cupy:
        cp.random.seed(seed)
        X = cupyx.scipy.sparse.random(
            n_samples, n_features, density=density, format="csr", dtype=dtype
        )
    else:
        np.random.seed(seed)
        X = scipy.sparse.random(
            n_samples, n_features, density=density, format="csr", dtype=dtype
        )
    X.data[:] = 1.0
    return X  # type: ignore[return-value]


def evaluate_recall_at_k(
    X: ArrayLike,
    Q: ArrayLike,
    indices: Union[np.ndarray, cp.ndarray],
    offsets: Union[np.ndarray, cp.ndarray],
    get_top_k_fn: Callable[
        [
            ArrayLike,
            ArrayLike,
            int,
            int,
        ],
        Union[np.ndarray, cp.ndarray],
    ],
    k: int,
) -> list[float]:
    """Evaluate recall@k for each query"""
    assert Q.shape is not None, "Shape is None"
    n_queries = Q.shape[0]
    recalls = []
    for q_idx in range(n_queries):
        start, end = int(offsets[q_idx]), int(offsets[q_idx + 1])
        lsh_indices = indices[start:end]
        gt_indices = get_top_k_fn(X, Q, q_idx, k)
        recalls.append(compute_recall(lsh_indices, gt_indices))
    return recalls
