"""
Utility functions.
"""

from typing import Union

import cupy as cp
import cupyx.scipy.sparse
import numpy as np
import scipy.sparse


def get_array_info(
    arr: Union[
        np.ndarray, cp.ndarray, scipy.sparse.csr_matrix, cupyx.scipy.sparse.csr_matrix
    ],
) -> tuple[int, int, np.dtype]:
    """Get shape and dtype from numpy or cupy array."""
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr.ndim}D")
    assert arr.shape is not None, "Shape is None"
    return arr.shape[0], arr.shape[1], arr.dtype


def ensure_device_array(
    arr: Union[
        np.ndarray, cp.ndarray, scipy.sparse.csr_matrix, cupyx.scipy.sparse.csr_matrix
    ],
) -> Union[cp.ndarray, cupyx.scipy.sparse.csr_matrix]:
    """
    Ensure array is on GPU. Dense arrays become C-contiguous cupy arrays.
    Sparse CSR matrices become cupyx CSR matrices. If on host, copies to device.
    """
    if isinstance(arr, cupyx.scipy.sparse.csr_matrix):
        return arr
    if isinstance(arr, scipy.sparse.csr_matrix):
        return cupyx.scipy.sparse.csr_matrix(arr)

    if hasattr(arr, "__cuda_array_interface__"):
        if not arr.flags.c_contiguous:
            return cp.ascontiguousarray(arr)
        return arr
    else:
        return cp.asarray(arr, order="C")


def compute_recall(lsh_indices: np.ndarray, gt_indices: np.ndarray) -> float:
    """Compute recall score between LSH candidates and ground truth"""
    if len(gt_indices) == 0:
        return 0.0
    intersection = np.intersect1d(lsh_indices, gt_indices)
    return len(intersection) / len(gt_indices)
