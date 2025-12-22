"""
Utility functions.
"""

import cupy as cp
import numpy as np


def get_array_info(arr: np.ndarray | cp.ndarray) -> tuple[int, int, np.dtype]:
    """Get shape and dtype from numpy or cupy array."""
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr.ndim}D")
    return arr.shape[0], arr.shape[1], arr.dtype


def ensure_device_array(arr: np.ndarray | cp.ndarray) -> cp.ndarray:
    """
    Ensure array is on GPU as a C-contiguous cupy array.
    If on host, copies to device.
    """
    if hasattr(arr, "__cuda_array_interface__"):
        if not arr.flags.c_contiguous:
            return cp.ascontiguousarray(arr)
        return arr
    else:
        return cp.asarray(arr, order="C")
