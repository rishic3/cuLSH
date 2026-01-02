"""
p-Stable LSH
"""

from typing import Literal, Optional, Union

import cupy as cp
import numpy as np

from culsh._culsh_core import Candidates, PSLSHCore, PSLSHIndex
from culsh.utils import ensure_device_array, get_array_info, resolve_seed


class PStableLSH:
    """
    Locality sensitive hashing using p-stable distributions. This approximates Euclidean
    distance between vectors for ANN search.

    Parameters
    ----------
    n_hash_tables : int
        Number of hash tables (OR-amplification of the locality-sensitive family).
        More tables provide additional independent chances to find neighbors,
        improving recall at the cost of more false positives. Corresponds to 'b'
        in the amplified probability (1-(1-p_w^r)^b), where p_w is the probability of
        collision for window size w.
    n_hashes : int
        Number of hashes (random projections) per table (AND-amplification of the
        locality-sensitive family). More hashes require samples to agree on more hash bits,
        increasing precision at the cost of more false negatives. Corresponds to 'r' in the
        amplified probability (1-(1-p_w^r)^b), where p_w is the probability of
        collision for window size w.
    window_size : int or "auto"
        The quantization width for projections. Determines the resolution of the hash function
        by defining the physical size of the hash buckets. Larger window size increases the
        base collision probability. If "auto", the window size is estimated from data magnitude
        during fit() by sampling vectors and computing mean_magnitude / 2.
    seed : int, optional
        Random seed for reproducible hashes. If None (default), a random seed is used.

    Examples
    --------
    >>> import numpy as np
    >>> from culsh import PStableLSH
    >>>
    >>> # Create random data
    >>> X = np.random.randn(10000, 128).astype(np.float32)
    >>> Q = np.random.randn(100, 128).astype(np.float32)
    >>>
    >>> # Fit model
    >>> lsh = PStableLSH(n_hash_tables=16, n_hashes=8, window_size=4)
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
        window_size: Union[int, Literal["auto"]] = "auto",
        seed: Optional[int] = None,
    ):
        if n_hash_tables <= 0:
            raise ValueError("n_hash_tables must be positive")
        if n_hashes <= 0:
            raise ValueError("n_hashes must be positive")
        if window_size != "auto" and window_size <= 0:
            raise ValueError("window_size must be positive")

        self._n_hash_tables = n_hash_tables
        self._n_hashes = n_hashes
        self._window_size: Union[int, Literal["auto"]] = window_size
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
    def window_size(self) -> Union[int, Literal["auto"]]:
        """Window size or 'auto' if estimated at fit time."""
        return self._window_size

    @property
    def seed(self) -> int:
        """Random seed."""
        return self._seed

    def fit(self, X: Union[np.ndarray, cp.ndarray]) -> "PStableLSHModel":
        """
        Fit the PStableLSH model on input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input vectors. Can be numpy or cupy array.

        Returns
        -------
        PStableLSHModel
            The fitted model containing the LSH index.
        """
        n_samples, n_features, dtype = get_array_info(X)

        if self._window_size == "auto":
            window_size = self._estimate_window_size(X)
        else:
            window_size = self._window_size

        X = ensure_device_array(X)

        core = PSLSHCore(self._n_hash_tables, self._n_hashes, window_size, self._seed)

        if dtype == np.float32:
            index = core.fit_float(X, n_samples, n_features)
        elif dtype == np.float64:
            index = core.fit_double(X, n_samples, n_features)
        else:
            raise TypeError(
                f"Unsupported dtype: {dtype}. Supported dtypes are float32, float64."
            )

        return PStableLSHModel(
            n_hash_tables=self._n_hash_tables,
            n_hashes=self._n_hashes,
            window_size=window_size,
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

        if self._window_size == "auto":
            window_size = self._estimate_window_size(X)
        else:
            window_size = self._window_size

        X = ensure_device_array(X)

        core = PSLSHCore(self._n_hash_tables, self._n_hashes, window_size, self._seed)

        if dtype == np.float32:
            return core.fit_query_float(X, n_samples, n_features)
        elif dtype == np.float64:
            return core.fit_query_double(X, n_samples, n_features)
        else:
            raise TypeError(
                f"Unsupported dtype: {dtype}. Supported dtypes are float32, float64."
            )

    @staticmethod
    def _estimate_window_size(
        X: Union[np.ndarray, cp.ndarray],
        scale_factor: float = 2.0,
        max_samples: int = 10000,
    ) -> int:
        """
        Sample up to max_samples vectors and compute the mean magnitude, then divide by
        scale_factor to estimate window size.

        Parameters
        ----------
        X : array-like
            Input data (numpy or cupy).
        scale_factor : float
            Divisor for mean magnitude. Smaller values = larger window = more collisions.
        max_samples : int
            Maximum number of samples to use for estimation.

        Returns
        -------
        int
            Estimated window size (minimum 1).
        """
        xp = cp if isinstance(X, cp.ndarray) else np

        n_samples = len(X)
        if n_samples > max_samples:
            indices = xp.random.choice(n_samples, max_samples, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X

        mean_magnitude = float(xp.linalg.norm(X_sample, axis=1).mean())
        return max(1, int(mean_magnitude / scale_factor))


class PStableLSHModel:
    """
    Model produced by PStableLSH.fit() containing the fitted PStableLSH index.

    Parameters
    ----------
    n_hash_tables : int
        Number of hash tables.
    n_hashes : int
        Number of hashes per hash table.
    n_features : int
        Number of features.
    window_size : int
        Window size.
    core : PSLSHCore
        Core PStableLSH object containing the fitted index.
    index : PSLSHIndex
        Fitted index.
    """

    def __init__(
        self,
        n_hash_tables: int,
        n_hashes: int,
        window_size: int,
        n_features: int,
        core: PSLSHCore,
        index: PSLSHIndex,
    ):
        self._n_hash_tables = n_hash_tables
        self._n_hashes = n_hashes
        self._window_size = window_size
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
    def window_size(self) -> int:
        """Window size."""
        return self._window_size

    @property
    def n_features(self) -> int:
        """Number of features."""
        return self._n_features

    @property
    def index(self) -> PSLSHIndex:
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
        Save the PStableLSH model to a file.

        Parameters
        ----------
        path : str
            Path to save the model.
        """
        np.savez_compressed(
            path,
            n_hash_tables=self._n_hash_tables,
            n_hashes=self._n_hashes,
            window_size=self._window_size,
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
            bias=self._index.get_bias(),
        )

    @classmethod
    def load(cls, path: str) -> "PStableLSHModel":
        """
        Load the PStableLSH model from a file.

        Parameters
        ----------
        path : str
            Path to load the model from.
        """
        data = np.load(path)

        core = PSLSHCore(
            n_hash_tables=int(data["n_hash_tables"]),
            n_hashes=int(data["n_hashes"]),
            window_size=int(data["window_size"]),
            seed=int(data["seed"]),
        )

        index_kwargs = {
            "candidate_indices": data["candidate_indices"],
            "bucket_signatures": data["bucket_signatures"],
            "bucket_candidate_offsets": data["bucket_candidate_offsets"],
            "table_bucket_offsets": data["table_bucket_offsets"],
            "projection": data["projection"],
            "bias": data["bias"],
            "n_total_candidates": int(data["n_total_candidates"]),
            "n_total_buckets": int(data["n_total_buckets"]),
            "n_hash_tables": int(data["n_hash_tables"]),
            "n_hashes": int(data["n_hashes"]),
            "sig_nbytes": int(data["sig_nbytes"]),
            "n_features": int(data["n_features"]),
            "seed": int(data["seed"]),
        }

        if data["is_double"]:
            index = PSLSHIndex.load_double(**index_kwargs)
        else:
            index = PSLSHIndex.load_float(**index_kwargs)

        return cls(
            n_hash_tables=int(data["n_hash_tables"]),
            n_hashes=int(data["n_hashes"]),
            window_size=int(data["window_size"]),
            n_features=int(data["n_features"]),
            core=core,
            index=index,
        )
