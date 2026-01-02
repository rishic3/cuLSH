"""
Reference CPU implementations of LSH algorithms for testing.
"""

import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable, Optional

import numpy as np
import scipy.sparse as sp
from typing_extensions import override

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LSH(ABC):
    """
    Base class for reference LSH algorithms
    """

    def __init__(self, n_hash_tables: int, n_hashes: int, seed: Optional[int] = None):
        self._n_hash_tables = n_hash_tables
        self._n_hashes = n_hashes
        self._seed = seed

    @property
    def n_hash_tables(self) -> int:
        """Number of hash tables."""
        return self._n_hash_tables

    @property
    def n_hashes(self) -> int:
        """Number of hash functions per hash table."""
        return self._n_hashes

    @property
    def seed(self) -> Optional[int]:
        """Random seed."""
        return self._seed

    @abstractmethod
    def _hash(self, X, H) -> np.ndarray:
        """Hash the input data X"""
        raise NotImplementedError("Subclasses must implement _hash")

    @abstractmethod
    def _generate_hash_params(self, n_features: int, dtype: Any) -> Any:
        """Generate parameters used for hashing"""
        raise NotImplementedError("Subclasses must implement _generate_hash_params")

    def fit(self, X) -> "LSHModel":
        """
        Fit the LSH model.

        Args:
            X: the n x d input matrix, where each row is a vector
        Returns:
            LSHModel: the fitted model
        """
        start_time = time.perf_counter()
        dtype = X.dtype
        hash_params = self._generate_hash_params(X.shape[1], dtype)
        H_x = self._hash(X, hash_params)

        # index is a list of dicts for each hash table
        # each dict maps the hash table signature to a list of vector indices in X
        index: list[dict[tuple[int, ...], list[int]]] = [
            defaultdict(list) for _ in range(self._n_hash_tables)
        ]

        for i, signature in enumerate(H_x):
            for j in range(self._n_hash_tables):
                table_start = j * self._n_hashes
                table_end = (j + 1) * self._n_hashes
                table_signature = signature[table_start:table_end]

                # convert signature to hashable type (tuple)
                index[j][tuple(table_signature)].append(i)

        end_time = time.perf_counter()
        logger.info("Fit completed in %s seconds", round(end_time - start_time, 2))

        return LSHModel(
            n_hash_tables=self._n_hash_tables,
            n_hashes=self._n_hashes,
            n_features=X.shape[1],
            index=index,
            hash_params=hash_params,
            hash_fn=self._hash,
        )


class LSHModel:
    """
    Model produced by LSH.fit() containing the fitted LSH index.
    """

    def __init__(
        self,
        n_hash_tables: int,
        n_hashes: int,
        n_features: int,
        index: Any,
        hash_params: Any,
        hash_fn: Callable[[Any, Any], np.ndarray],
    ):
        self._n_hash_tables = n_hash_tables
        self._n_hashes = n_hashes
        self._n_features = n_features
        self._index = index
        self._hash_params = hash_params
        self._hash_fn = hash_fn

    @property
    def n_hash_tables(self) -> int:
        """Number of hash tables."""
        return self._n_hash_tables

    @property
    def n_hashes(self) -> int:
        """Number of hash functions per hash table."""
        return self._n_hashes

    @property
    def n_features(self) -> int:
        """Number of features."""
        return self._n_features

    @property
    def index(self) -> Any:
        """Fitted index."""
        return self._index

    def query(self, Q) -> list[list[int]]:
        """
        Find the approximate nearest neighbors for the matrix of query vectors Q.

        Args:
            Q: the m x d query matrix, where each row is a vector
        Returns:
            list[list[int]]: list of candidate neighbors indices for each query
        """
        start_time = time.perf_counter()
        index = self._index

        H_q = self._hash_fn(Q, self._hash_params)
        all_candidates = []

        for q_signature in H_q:
            q_candidates = set()

            # for each hash table, retrieve candidates that hashed to that table from the index
            for j in range(self._n_hash_tables):
                table_start = j * self._n_hashes
                table_end = (j + 1) * self._n_hashes
                q_table_signature = q_signature[table_start:table_end]

                # get candidates from hash table j
                table_candidates = index[j].get(tuple(q_table_signature), [])
                if table_candidates:
                    q_candidates.update(table_candidates)

            if q_candidates:
                all_candidates.append(list(q_candidates))
            else:
                all_candidates.append([])

        end_time = time.perf_counter()
        logger.info("Query completed in %s seconds", round(end_time - start_time, 2))

        return all_candidates


class RandomProjectionLSH(LSH):
    """Reference random projection LSH implementation"""

    def __init__(self, n_hash_tables: int, n_projections: int, seed: int):
        super().__init__(n_hash_tables, n_projections, seed)

    @override
    def _generate_hash_params(self, n_features: int, dtype: Any) -> np.ndarray:
        """Generate a random projection matrix"""
        if self._seed is not None:
            np.random.seed(self._seed)
        return np.random.randn(self._n_hash_tables * self._n_hashes, n_features).astype(
            dtype
        )

    @override
    def _hash(self, X, H) -> np.ndarray:
        """Hash input using sign of random projections"""
        return (X @ H.T > 0).astype(int)


MINHASH_PRIME = 4294967291


class MinHashLSH(LSH):
    """Reference MinHash LSH implementation"""

    def __init__(self, n_hash_tables: int, n_hashes: int, seed: int):
        super().__init__(n_hash_tables, n_hashes, seed)

    @override
    def _generate_hash_params(
        self, n_features: int, dtype: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate random integers A, B for universal hash functions"""
        if self._seed is not None:
            np.random.seed(self._seed)
        n_total = self._n_hash_tables * self._n_hashes
        A = np.random.randint(1, MINHASH_PRIME, size=n_total, dtype=np.uint64)
        B = np.random.randint(0, MINHASH_PRIME, size=n_total, dtype=np.uint64)
        return (A, B)

    @override
    def _hash(self, X, H) -> np.ndarray:
        """Compute MinHash signatures: min((A * idx + B) mod p) over non-zero indices"""
        A, B = H
        n_samples = X.shape[0]
        n_total = self._n_hash_tables * self._n_hashes

        if not sp.isspmatrix_csr(X):
            X = sp.csr_matrix(X)

        signatures = np.zeros((n_samples, n_total), dtype=np.uint32)

        for i in range(n_samples):
            row_start = X.indptr[i]
            row_end = X.indptr[i + 1]
            indices = X.indices[row_start:row_end].astype(np.uint64)

            if len(indices) == 0:
                signatures[i, :] = 0xFFFFFFFF
            else:
                hashes = (A[:, None] * indices[None, :] + B[:, None]) % MINHASH_PRIME
                signatures[i, :] = hashes.min(axis=1).astype(np.uint32)

        return signatures


class PStableLSH(LSH):
    """Reference P-Stable LSH implementation"""

    def __init__(
        self, n_hash_tables: int, n_hashes: int, window_size: float, seed: int
    ):
        super().__init__(n_hash_tables, n_hashes, seed)
        self._window_size = window_size

    @property
    def window_size(self) -> float:
        """Bucket width parameter."""
        return self._window_size

    @override
    def _generate_hash_params(
        self, n_features: int, dtype: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate random projection matrix P, bias terms b, and window size"""
        if self._seed is not None:
            np.random.seed(self._seed)
        n_total = self._n_hash_tables * self._n_hashes
        P = np.random.randn(n_total, n_features).astype(dtype)
        b = np.random.uniform(0, self._window_size, size=n_total).astype(dtype)
        w = np.array(self._window_size, dtype=dtype)
        return (P, b, w)

    @override
    def _hash(self, X, H) -> np.ndarray:
        """Compute hash h(x) = floor((P * x + b) / w)"""
        P, b, w = H
        projections = X @ P.T + b
        return np.floor(projections / w).astype(np.int32)
