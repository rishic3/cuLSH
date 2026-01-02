from typing import Any, Callable, Union

import numpy as np
import scipy.sparse

from culsh.utils import compute_recall


def generate_dense_data(
    n_samples: int, n_features: int, dtype: Any = np.float32, seed: int = 42
) -> np.ndarray:
    """Generate random dense matrix"""
    np.random.seed(seed)
    return np.random.randn(n_samples, n_features).astype(dtype)


def generate_sparse_data(
    n_samples: int,
    n_features: int,
    density: float,
    dtype: Any = np.float32,
    seed: int = 42,
) -> scipy.sparse.csr_matrix:
    """Generate random sparse binary CSR matrix"""
    np.random.seed(seed)
    X = scipy.sparse.random(
        n_samples, n_features, density=density, format="csr", dtype=dtype
    )
    X.data[:] = 1.0
    return X  # type: ignore[return-value]


def evaluate_recall_at_k(
    X: Union[np.ndarray, scipy.sparse.csr_matrix],
    Q: Union[np.ndarray, scipy.sparse.csr_matrix],
    indices: np.ndarray,
    offsets: np.ndarray,
    get_top_k_fn: Callable[
        [
            Union[np.ndarray, scipy.sparse.csr_matrix],
            Union[np.ndarray, scipy.sparse.csr_matrix],
            int,
            int,
        ],
        np.ndarray,
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
