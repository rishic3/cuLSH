"""
Random Projection LSH
"""

from typing import Optional, Union

import cupy as cp
import numpy as np

from culsh._culsh_core import Candidates, Index, RPLSHCore
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
        in the amplified probability (1 - (1 - p^r)^b).
    n_projections : int
        Number of projections per table (AND-amplification of the locality-sensitive family).
        More projections require samples to agree on more hash bits, increasing precision
        at the cost of more false negatives. Corresponds to 'r' in the amplified
        probability (1 - (1 - p^r)^b).
    seed : int, optional
        Random seed for reproducible projections. Default is 42.

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
    >>> lsh = RPLSH(n_hash_tables=16, n_projections=8)
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
        n_projections: int = 8,
        seed: int = 42,
    ):
        if n_hash_tables <= 0:
            raise ValueError("n_hash_tables must be positive")
        if n_projections <= 0:
            raise ValueError("n_projections must be positive")

        self._n_hash_tables = n_hash_tables
        self._n_projections = n_projections
        self._seed = seed

    @property
    def n_hash_tables(self) -> int:
        """Number of hash tables."""
        return self._n_hash_tables

    @property
    def n_projections(self) -> int:
        """Number of projections per hash table."""
        return self._n_projections

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

        core = RPLSHCore(self._n_hash_tables, self._n_projections, self._seed)

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
            n_projections=self._n_projections,
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

        core = RPLSHCore(self._n_hash_tables, self._n_projections, self._seed)

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
    n_projections : int
        Number of projections per hash table.
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
        n_projections: int,
        n_features: int,
        core: RPLSHCore,
        index: Index,
    ):
        self._n_hash_tables = n_hash_tables
        self._n_projections = n_projections
        self._n_features = n_features
        self._core = core
        self._index = index

    @property
    def n_hash_tables(self) -> int:
        """Number of hash tables."""
        return self._n_hash_tables

    @property
    def n_projections(self) -> int:
        """Number of projections per hash table."""
        return self._n_projections

    @property
    def n_features(self) -> int:
        """Number of features."""
        return self._n_features

    @property
    def index(self) -> Index:
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
