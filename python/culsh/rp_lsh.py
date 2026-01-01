"""
Random Projection LSH
"""

from typing import Optional, Union

import cupy as cp
import numpy as np

from culsh._culsh_core import Candidates, RPLSHCore, RPLSHIndex
from culsh.utils import ensure_device_array, get_array_info


class RPLSH:
    """
    Locality sensitive hashing using random projections. This approximates cosine distance
    between vectors for ANN search.

    Parameters
    ----------
    n_hash_tables : int
        Number of hash tables (OR-amplification of the locality-sensitive family).
        More tables provide additional independent chances to find neighbors,
        improving recall at the cost of more false positives. Corresponds to 'b'
        in the amplified probability (1-(1-s^r)^b), where s is the cosine similarity
        between two vectors.
    n_hashes : int
        Number of hashes (random projections) per table (AND-amplification of the
        locality-sensitive family). More hashes require samples to agree on more hash bits,
        increasing precision at the cost of more false negatives. Corresponds to 'r' in the
        amplified probability (1-(1-s^r)^b), where s is the cosine similarity between two
        vectors.
    seed : int, optional
        Random seed for reproducible hashes. Default is 42.

    Examples
    --------
    >>> import numpy as np
    >>> from culsh import RPLSH
    >>>
    >>> # Create random data
    >>> X = np.random.randn(10000, 128).astype(np.float32)
    >>> Q = np.random.randn(100, 128).astype(np.float32)
    >>>
    >>> # Fit model
    >>> lsh = RPLSH(n_hash_tables=16, n_hashes=8)
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
        n_hash_tables: int = 16,
        n_hashes: int = 8,
        seed: int = 42,
    ):
        if n_hash_tables <= 0:
            raise ValueError("n_hash_tables must be positive")
        if n_hashes <= 0:
            raise ValueError("n_hashes must be positive")

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
    def seed(self) -> int:
        """Random seed."""
        return self._seed

    def fit(self, X: Union[np.ndarray, cp.ndarray]) -> "RPLSHModel":
        """
        Fit the RPLSH model on input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input vectors. Can be numpy or cupy array.

        Returns
        -------
        RPLSHModel
            The fitted model containing the LSH index.
        """
        n_samples, n_features, dtype = get_array_info(X)
        X = ensure_device_array(X)

        core = RPLSHCore(self._n_hash_tables, self._n_hashes, self._seed)

        if dtype == np.float32:
            index = core.fit_float(X, n_samples, n_features)
        elif dtype == np.float64:
            index = core.fit_double(X, n_samples, n_features)
        else:
            raise TypeError(
                f"Unsupported dtype: {dtype}. Supported dtypes are float32, float64."
            )

        return RPLSHModel(
            n_hash_tables=self._n_hash_tables,
            n_hashes=self._n_hashes,
            n_features=n_features,
            core=core,
            index=index,
        )

    def fit_query(
        self, X: Union[np.ndarray, cp.ndarray], batch_size: Optional[int] = None
    ) -> Candidates:
        """
        Simultaneously fit and query the LSH index. This is more efficient than
        calling fit() followed by query() when querying the same data used for fitting.
        Note: input vectors are considered candidate neighbors of themselves.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input vectors to fit and query. Can be numpy or cupy array.
        batch_size : int, optional
            If specified, falls back to fit() + query() with batching to reduce
            peak memory usage.

        Returns
        -------
        Candidates
            Query results containing candidate indices for each sample.
        """
        if batch_size is not None:
            model = self.fit(X)
            return model.query(X, batch_size=batch_size)

        n_samples, n_features, dtype = get_array_info(X)
        X = ensure_device_array(X)

        core = RPLSHCore(self._n_hash_tables, self._n_hashes, self._seed)

        if dtype == np.float32:
            return core.fit_query_float(X, n_samples, n_features)
        elif dtype == np.float64:
            return core.fit_query_double(X, n_samples, n_features)
        else:
            raise TypeError(
                f"Unsupported dtype: {dtype}. Supported dtypes are float32, float64."
            )


class RPLSHModel:
    """
    Model produced by RPLSH.fit() containing the fitted LSH index.

    Parameters
    ----------
    n_hash_tables : int
        Number of hash tables.
    n_hashes : int
        Number of hashes per hash table.
    n_features : int
        Number of features.
    core : RPLSHCore
        Core RPLSH object containing the fitted index.
    index : Index
        Fitted index.
    """

    def __init__(
        self,
        n_hash_tables: int,
        n_hashes: int,
        n_features: int,
        core: RPLSHCore,
        index: RPLSHIndex,
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
        """Number of hashes per hash table."""
        return self._n_hashes

    @property
    def n_features(self) -> int:
        """Number of features."""
        return self._n_features

    @property
    def index(self) -> RPLSHIndex:
        """The fitted index."""
        return self._index

    def query(
        self, Q: Union[np.ndarray, cp.ndarray], batch_size: Optional[int] = None
    ) -> Candidates:
        """
        Find candidate neighbors for the query vectors Q.

        Parameters
        ----------
        Q : array-like of shape (n_queries, n_features)
            Query vectors. Can be numpy or cupy array.
        batch_size : int, optional
            If specified, process queries in batches of this size to reduce
            peak memory usage.

        Returns
        -------
        Candidates
            Query results containing candidate indices for each query.
        """
        n_queries, n_features, dtype = get_array_info(Q)
        Q = ensure_device_array(Q)

        if n_features != self._n_features:
            raise ValueError(
                f"Query features ({n_features}) != fitted features ({self._n_features})"
            )

        if dtype == np.float32:
            return self._core.query_float(Q, n_queries, self._index, batch_size)
        elif dtype == np.float64:
            return self._core.query_double(Q, n_queries, self._index, batch_size)
        else:
            raise TypeError(f"Unsupported dtype: {dtype}. Use float32 or float64.")

    def save(self, path: str) -> None:
        """
        Save the RPLSH model to a file.

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
            is_double=self._index.is_double,
            candidate_indices=self._index.get_candidate_indices(),
            bucket_signatures=self._index.get_bucket_signatures(),
            bucket_candidate_offsets=self._index.get_bucket_candidate_offsets(),
            table_bucket_offsets=self._index.get_table_bucket_offsets(),
            projection=self._index.get_projection_matrix(),
        )

    @classmethod
    def load(cls, path: str) -> "RPLSHModel":
        """
        Load the RPLSH model from a file.

        Parameters
        ----------
        path : str
            Path to load the model from.
        """
        data = np.load(path)

        core = RPLSHCore(
            n_hash_tables=int(data["n_hash_tables"]),
            n_hashes=int(data["n_hashes"]),
            seed=int(data["seed"]),
        )

        index_kwargs = {
            "candidate_indices": data["candidate_indices"],
            "bucket_signatures": data["bucket_signatures"],
            "bucket_candidate_offsets": data["bucket_candidate_offsets"],
            "table_bucket_offsets": data["table_bucket_offsets"],
            "projection": data["projection"],
            "n_total_candidates": int(data["n_total_candidates"]),
            "n_total_buckets": int(data["n_total_buckets"]),
            "n_hash_tables": int(data["n_hash_tables"]),
            "n_hashes": int(data["n_hashes"]),
            "sig_nbytes": int(data["sig_nbytes"]),
            "n_features": int(data["n_features"]),
            "seed": int(data["seed"]),
        }

        if data["is_double"]:
            index = RPLSHIndex.load_double(**index_kwargs)
        else:
            index = RPLSHIndex.load_float(**index_kwargs)

        return cls(
            n_hash_tables=int(data["n_hash_tables"]),
            n_hashes=int(data["n_hashes"]),
            n_features=int(data["n_features"]),
            core=core,
            index=index,
        )
