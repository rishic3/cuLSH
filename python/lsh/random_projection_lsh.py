import logging
import os
import pickle
import time
from collections import defaultdict
from typing import Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RandomProjectionLSH:
    """
    Random projection LSH approximates cosine distance between vectors for ANN search.
    """

    def __init__(
        self,
        n_hash_tables: int,
        n_projections: int,
        store_data: Optional[bool] = False,
        seed: Optional[int] = None,
    ):
        """
        Initialize the RandomProjectionLSH class.

        Args:
            n_hash_tables: The number of hash tables. This parameter corresponds to an OR-amplification
                           of the locality-sensitive family. A higher value increases the probability of
                           finding a candidate neighbor. Corresponds to 'b' in the amplified probability
                           (1 - (1 - p^r)^b).
            n_projections: The number of random hyperplanes (hash functions) per hash table. This parameter
                           corresponds to an AND amplification of the locality-sensitive family. A higher value
                           decreases the probability of finding a candidate neighbor. Corresponds to 'r'
                           in the amplified probability (1 - (1 - p^r)^b).
            store_data: If enabled, store the input vectors in the resultant model. The subsequent LSH
                        model will the vectors in the original dataset rather than just the vector indices.
                        Disabled by default.
            seed: Optional seed used to generate random projections, default is None.

        """
        self._n_hash_tables = n_hash_tables
        self._n_projections = n_projections
        self._n_hash = n_hash_tables * n_projections
        self._store_data = store_data
        self._seed = seed

    @property
    def n_hash_tables(self):
        return self._n_hash_tables

    @property
    def n_projections(self):
        return self._n_projections

    @property
    def store_data(self):
        return self._store_data

    @property
    def seed(self):
        return self._seed

    def _generate_random_projections(self, n_hash: int, d: int) -> np.ndarray:
        """
        Sample n_hash random unit vectors from a d-dimensional sphere.

        Args:
            n_hash: Number of vectors to sample
            d: Dimensionality of each vector
        Returns:
            np.ndarray: n_hash x d matrix of random unit vectors
        """
        if self._seed is not None:
            np.random.seed(self._seed)

        random_vecs = np.random.randn(n_hash, d)
        norms = np.linalg.norm(random_vecs, axis=1, keepdims=True)
        return random_vecs / norms

    def _hash(self, X: np.ndarray, P: np.ndarray) -> np.ndarray:
        """
        Hash an n x d input matrix X using the n_hash x d matrix of normal unit vectors P.

        Args:
            X: n x d input matrix, where each row is a vector
            P: n_hash x d matrix of normal unit vectors
        Returns:
            np.ndarray: n x n_hash matrix of signature bit vectors
        """
        return (X @ P.T > 0).astype(int)

    def fit(self, X: np.ndarray) -> "RandomProjectionLSHModel":
        """
        Fit the RandomProjectionLSH model.

        Args:
            X: the n x d input matrix, where each row is a vector
        Returns:
            RandomProjectionLSHModel: the fitted model
        """
        start_time = time.time()
        d = X.shape[1]
        P = self._generate_random_projections(self._n_hash, d)
        H_x = self._hash(X, P)

        # index is a list of dicts for each hash table
        # each dict maps the hash table signature to a list of vector indices in X
        index: list[dict[tuple[int, ...], list[int]]] = [
            defaultdict(list) for _ in range(self._n_hash_tables)
        ]

        for i, signature in enumerate(H_x):
            for j in range(self._n_hash_tables):
                table_start = j * self._n_projections
                table_end = (j + 1) * self._n_projections
                table_signature = signature[table_start:table_end]

                # convert signature to hashable type (tuple)
                index[j][tuple(table_signature)].append(i)

        end_time = time.time()
        logger.info("Fit completed in %s seconds", round(end_time - start_time, 2))

        if self._store_data:
            return RandomProjectionLSHModel(
                self._n_hash_tables, self._n_projections, index, P, X
            )
        else:
            return RandomProjectionLSHModel(
                self._n_hash_tables, self._n_projections, index, P
            )


class RandomProjectionLSHModel:
    """Model produced by RandomProjectionLSH.fit()"""

    def __init__(
        self,
        n_hash_tables: int,
        n_projections: int,
        index: list[dict[tuple[int, ...], list[int]]],
        P: np.ndarray,
        X: Optional[np.ndarray] = None,
    ):
        """
        Initialize the RandomProjectionLSHModel.

        Args:
            index: the index of the input vectors
            P: the n_hash x d matrix of normal unit vectors
            d: the dimensionality of the input vectors
            X: the input vectors if store_data is disabled
        """
        self._n_hash_tables = n_hash_tables
        self._n_projections = n_projections
        self._index = index
        self._P = P
        self._X = X

    @property
    def n_hash_tables(self):
        return self._n_hash_tables

    @property
    def n_projections(self):
        return self._n_projections

    @property
    def store_data(self):
        return self._X is not None

    def _hash(self, Q: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Hash the query matrix Q using the matrix of normal unit vectors P."""
        return (Q @ P.T > 0).astype(int)

    def query(self, Q: np.ndarray) -> list[list[int]] | list[list[np.ndarray]]:
        """
        Find the approximate nearest neighbors for the matrix of query vectors Q.

        Args:
            Q: the m x d query matrix, where each row is a vector
            k: the number of nearest neighbors to find
        Returns:
            list[list[int]]: If store_data=False, list of candidate neighbors indices for each query
            or
            list[list[np.ndarray]]: list of candidate neighbor vectors for each query
        """
        start_time = time.time()
        index = self._index

        H_q = self._hash(Q, self._P)
        all_candidates = []

        for q_signature in H_q:
            q_candidates = set()

            # for each hash table, retrieve candidates that hashed to that table from the index
            for j in range(self._n_hash_tables):
                table_start = j * self._n_projections
                table_end = (j + 1) * self._n_projections
                q_table_signature = q_signature[table_start:table_end]

                # get candidates from hash table j
                table_candidates = index[j].get(tuple(q_table_signature), [])
                if table_candidates:
                    q_candidates.update(table_candidates)

            if q_candidates:
                if self._X is not None:
                    all_candidates.append([self._X[i] for i in list(q_candidates)])
                else:
                    all_candidates.append(list(q_candidates))
            else:
                all_candidates.append([])

        end_time = time.time()
        logger.info("Query completed in %s seconds", round(end_time - start_time, 2))

        return all_candidates

    def save(self, save_dir: str):
        """Write the model to save dir."""
        os.makedirs(save_dir, exist_ok=True)

        arrs_to_save = {"P": self._P}
        if self._X is not None:
            arrs_to_save["X"] = self._X

        npz_path = os.path.join(save_dir, "arrays.npz")
        np.savez_compressed(npz_path, **arrs_to_save)

        attr_path = os.path.join(save_dir, "attributes.pkl")
        attrs = {
            "n_hash_tables": self._n_hash_tables,
            "n_projections": self._n_projections,
            "index": self._index,
        }

        with open(attr_path, "wb") as f:
            pickle.dump(attrs, f)

        logger.info("Model saved to %s", save_dir)

    @classmethod
    def load(cls, save_dir: str):
        """Read the model from save dir."""
        npz_path = os.path.join(save_dir, "arrays.npz")
        with np.load(npz_path) as data:
            P = data["P"]
            X = data.get("X")

        attr_path = os.path.join(save_dir, "attributes.pkl")
        with open(attr_path, "rb") as f:
            attrs = pickle.load(f)

        clazz = cls(
            n_hash_tables=attrs["n_hash_tables"],
            n_projections=attrs["n_projections"],
            index=attrs["index"],
            P=P,
            X=X,
        )
        logger.info("Model loaded from %s", save_dir)

        return clazz
