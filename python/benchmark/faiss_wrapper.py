"""
FAISS wrapper to match cuLSH fit/query API for benchmarking.
"""

import faiss
import numpy as np


class FaissCandidates:
    """FAISS candidates wrapper matching cuLSH Candidates"""

    def __init__(self, indices: np.ndarray):
        """
        Parameters
        ----------
        indices : np.ndarray of shape (n_queries, k)
            FAISS search results (top-k indices per query)
        """
        n_queries, k = indices.shape
        # Flatten to match cuLSH format
        self._indices = indices.ravel().astype(np.int32)
        # Each query has exactly k candidates
        self._offsets = np.arange(0, (n_queries + 1) * k, k, dtype=np.int64)

    @property
    def n_queries(self) -> int:
        return len(self._offsets) - 1

    @property
    def n_total_candidates(self) -> int:
        return len(self._indices)

    def get_indices(self):
        return self._indices

    def get_offsets(self):
        return self._offsets


class FaissIndexWrapper:
    """Wrapper for FAISS index matching cuLSH Index"""

    def __init__(self, index: faiss.IndexLSH, nbits: int):
        self._index = index
        self._nbits = nbits

    def size_bytes(self) -> int:
        """Estimate index size in bytes."""
        code_size = (self._nbits + 7) // 8
        return self._index.ntotal * code_size


class FaissLSHWrapper:
    """FAISS IndexLSH wrapper matching cuLSH API"""

    def __init__(
        self,
        n_hash_tables: int,
        n_hashes: int,
        n_candidates: int = 1000,
    ):
        """
        Parameters
        ----------
        n_hash_tables : int
            Number of hash tables (used to compute nbits)
        n_hashes : int
            Number of hashes per table (used to compute nbits)
        n_candidates : int
            Number of candidates to return per query (FAISS returns fixed k)
        """
        self.n_hash_tables = n_hash_tables
        self.n_hashes = n_hashes
        self.nbits = n_hash_tables * n_hashes
        self.n_candidates = n_candidates
        self._index = None
        self._d = None

    @property
    def index(self) -> FaissIndexWrapper:
        assert self._index is not None, "Index not fitted"
        return FaissIndexWrapper(self._index, self.nbits)

    def fit(self, X: np.ndarray):
        """Build the FAISS LSH index."""
        X = np.ascontiguousarray(X, dtype=np.float32)
        self._d = X.shape[1]

        self._index = faiss.IndexLSH(self._d, self.nbits)
        self._index.add(X)  # type: ignore
        return self

    def query(self, Q: np.ndarray) -> FaissCandidates:
        """Query the FAISS index."""
        assert self._index is not None, "Index not fitted"
        Q = np.ascontiguousarray(Q, dtype=np.float32)

        # returns (distances, indices)
        _, indices = self._index.search(Q, self.n_candidates)  # type: ignore
        return FaissCandidates(indices)

    def fit_query(self, X: np.ndarray) -> FaissCandidates:
        """Fit and query the input matrix."""
        self.fit(X)
        return self.query(X)
