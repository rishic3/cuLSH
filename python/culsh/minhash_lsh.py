"""
MinHash LSH
"""

from typing import Optional, Union

import cupy as cp
import cupyx.scipy.sparse
import numpy as np
import scipy.sparse

from culsh._culsh_core import Candidates, MinHashCore, MinHashIndex
from culsh.utils import ensure_device_array, get_array_info, resolve_seed


class MinHashLSH:
    """
    Locality sensitive hashing using min-wise independent permutations. This approximates
    Jaccard similarity between sets for ANN search.

    Parameters
    ----------
    n_hash_tables : int
        Number of hash tables (OR-amplification of the locality-sensitive family).
        More tables provide additional independent chances to find neighbors,
        improving recall at the cost of more false positives. Corresponds to 'b'
        in the amplified probability of collision (1-(1-s^r)^b), where s is the
        Jaccard similarity between two sets.
    n_hashes : int
        Number of hash functions per table (AND-amplification of the locality-sensitive
        family). More hashes require samples to agree on more hash bits, increasing
        precision at the cost of more false negatives. Corresponds to 'r' in the amplified
        probability of collision (1-(1-s^r)^b), where s is the Jaccard similarity between
        two sets.
    seed : int, optional
        Random seed for reproducible hashes. If None (default), a random seed is used.

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.sparse
    >>> from culsh import MinHashLSH
    >>>
    >>> # Create random sparse data
    >>> X = scipy.sparse.random(10000, 1000, density=0.01, format='csr')
    >>> Q = scipy.sparse.random(100, 1000, density=0.01, format='csr')
    >>>
    >>> # Fit model
    >>> lsh = MinHashLSH(n_hash_tables=16, n_hashes=8)
    >>> model = lsh.fit(X)
    >>>
    >>> # Query for candidates
    >>> candidates = model.query(Q)
    >>> indices = candidates.get_indices()
    >>> counts = candidates.get_counts()
    >>> offsets = candidates.get_offsets()
    """

    def __init__(
        self,
        n_hash_tables: int,
        n_hashes: int,
        seed: Optional[int] = None,
    ):
        if n_hash_tables <= 0:
            raise ValueError("n_hash_tables must be positive")
        if n_hashes <= 0:
            raise ValueError("n_hashes must be positive")

        self._n_hash_tables = n_hash_tables
        self._n_hashes = n_hashes
        self._seed = resolve_seed(seed)

    @property
    def n_hash_tables(self) -> int:
        """Number of hash tables."""
        return self._n_hash_tables

    @property
    def n_hashes(self) -> int:
        """Number of hash functions per hash table."""
        return self._n_hashes

    @property
    def seed(self) -> int:
        """Random seed."""
        return self._seed

    def fit(
        self, X: Union[scipy.sparse.csr_matrix, cupyx.scipy.sparse.csr_matrix]
    ) -> "MinHashLSHModel":
        """
        Fit the MinHashLSH model on input data.

        Parameters
        ----------
        X : CSR matrix of shape (n_samples, n_features)
            Sparse input matrix. Can be scipy or cupyx CSR matrix.
            Note the matrix is assumed to represent binary set membership
            (magnitude of data values are ignored).

        Returns
        -------
        MinHashLSHModel
            The fitted model containing the LSH index.
        """
        n_samples, n_features, _ = get_array_info(X)
        X = ensure_device_array(X)

        indices, indptr = ensure_int32_indices(X)
        core = MinHashCore(self._n_hash_tables, self._n_hashes, self._seed)

        index = core.fit(indices, indptr, n_samples, n_features)

        return MinHashLSHModel(
            n_hash_tables=self._n_hash_tables,
            n_hashes=self._n_hashes,
            n_features=n_features,
            core=core,
            index=index,
        )

    def fit_query(
        self, X: Union[scipy.sparse.csr_matrix, cupyx.scipy.sparse.csr_matrix]
    ) -> Candidates:
        """
        Simultaneously fit and query the LSH index. This is more efficient than
        calling fit() followed by query() when querying the same data used for fitting.
        Note: input vectors are considered candidate neighbors of themselves.

        Parameters
        ----------
        X : CSR matrix of shape (n_samples, n_features)
            Sparse input matrix. Can be scipy or cupyx CSR matrix.

        Returns
        -------
        Candidates
            Query results containing candidate indices for each sample.
        """
        n_samples, n_features, _ = get_array_info(X)
        X = ensure_device_array(X)

        indices, indptr = ensure_int32_indices(X)
        core = MinHashCore(self._n_hash_tables, self._n_hashes, self._seed)

        return core.fit_query(indices, indptr, n_samples, n_features)


class MinHashLSHModel:
    """
    Model produced by MinHashLSH.fit() containing the fitted LSH index.

    Parameters
    ----------
    n_hash_tables : int
        Number of hash tables.
    n_hashes : int
        Number of hash functions per hash table.
    n_features : int
        Number of features.
    core : MinHashCore
        Core MinHash object containing the fitted index.
    index : MinHashIndex
        Fitted index.
    """

    def __init__(
        self,
        n_hash_tables: int,
        n_hashes: int,
        n_features: int,
        core: MinHashCore,
        index: MinHashIndex,
    ):
        self._n_hash_tables = n_hash_tables
        self._n_hashes = n_hashes
        self._n_features = n_features
        self._core = core
        self._index = index

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
    def index(self) -> MinHashIndex:
        """The fitted index."""
        return self._index

    def query(
        self, Q: Union[scipy.sparse.csr_matrix, cupyx.scipy.sparse.csr_matrix]
    ) -> Candidates:
        """
        Find candidate neighbors for the query vectors Q.

        Parameters
        ----------
        Q : CSR matrix of shape (n_queries, n_features)
            Sparse query matrix. Can be scipy or cupyx CSR matrix.

        Returns
        -------
        Candidates
            Query results containing candidate indices for each query.
        """
        n_queries, n_features, _ = get_array_info(Q)
        Q = ensure_device_array(Q)

        if n_features != self._n_features:
            raise ValueError(
                f"Query features ({n_features}) != fitted features ({self._n_features})"
            )

        indices, indptr = ensure_int32_indices(Q)

        return self._core.query(indices, indptr, n_queries, self._index)

    def save(self, path: str) -> None:
        """
        Save the MinHashLSH model to an npz file.

        Parameters
        ----------
        path : str
            Path to save the model.
        """
        np.savez_compressed(
            path,
            n_hash_tables=self._n_hash_tables,
            n_hashes=self._n_hashes,
            n_features=self._n_features,
            n_total_candidates=self._index.n_total_candidates,
            n_total_buckets=self._index.n_total_buckets,
            seed=self._index.seed,
            sig_nbytes=self._index.sig_nbytes,
            candidate_indices=self._index.get_candidate_indices(),
            bucket_signatures=self._index.get_bucket_signatures(),
            bucket_candidate_offsets=self._index.get_bucket_candidate_offsets(),
            table_bucket_offsets=self._index.get_table_bucket_offsets(),
            hash_a=self._index.get_hash_a(),
            hash_b=self._index.get_hash_b(),
        )

    @classmethod
    def load(cls, path: str) -> "MinHashLSHModel":
        """
        Load the MinHashLSH model from an npz file.

        Parameters
        ----------
        path : str
            Path to load the model from.
        """
        data = np.load(path)

        core = MinHashCore(
            n_hash_tables=int(data["n_hash_tables"]),
            n_hashes=int(data["n_hashes"]),
            seed=int(data["seed"]),
        )

        index = MinHashIndex.load(
            candidate_indices=data["candidate_indices"],
            bucket_signatures=data["bucket_signatures"],
            bucket_candidate_offsets=data["bucket_candidate_offsets"],
            table_bucket_offsets=data["table_bucket_offsets"],
            hash_a=data["hash_a"],
            hash_b=data["hash_b"],
            n_total_candidates=int(data["n_total_candidates"]),
            n_total_buckets=int(data["n_total_buckets"]),
            n_hash_tables=int(data["n_hash_tables"]),
            n_hashes=int(data["n_hashes"]),
            sig_nbytes=int(data["sig_nbytes"]),
            n_features=int(data["n_features"]),
            seed=int(data["seed"]),
        )

        return cls(
            n_hash_tables=int(data["n_hash_tables"]),
            n_hashes=int(data["n_hashes"]),
            n_features=int(data["n_features"]),
            core=core,
            index=index,
        )


def ensure_int32_indices(
    M: Union[scipy.sparse.csr_matrix, cupyx.scipy.sparse.csr_matrix],
) -> tuple[Union[np.ndarray, cp.ndarray], Union[np.ndarray, cp.ndarray]]:
    """Ensure the indices of the matrix are int32"""
    if M.indices.dtype != np.int32:
        raise ValueError(
            f"CUDA MinHashLSH requires int32 indices, got {M.indices.dtype}"
        )

    indices = M.indices.astype(np.int32)
    indptr = M.indptr.astype(np.int32)
    return indices, indptr
