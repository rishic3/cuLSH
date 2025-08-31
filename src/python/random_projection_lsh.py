import os
import pickle
from collections import defaultdict
from typing import Optional

import numpy as np


class RandomProjectionLSH:
    """
    Random projection LSH approximates cosine distance between vectors for ANN search.
    """

    def __init__(
        self, n_hash_tables: int, n_projections: int, index_only: Optional[bool] = True
    ):
        """
        Initialize the RandomProjectionLSH class.

        Args:
            n_hash_tables: The number of hash tables. This parameter corresponds to an OR-amplification
                           of the original family. A higher value increases the probability of
                           finding a candidate neighbor. Corresponds to 'b' in the amplified probability
                           (1 - (1 - p^r)^b).
            n_projections: The number of random hyperplanes (hash functions) per hash table. This parameter
                           corresponds to an AND amplification of the original family. A higher value
                           decreases the probability of finding a candidate neighbor. Corresponds to 'r'
                           in the amplified probability (1 - (1 - p^r)^b).
            index_only: If enabled, only store the LSH index and not the input vectors. The subsequent LSH
                        model will return the vector indices in the original dataset rather than the vectors
                        themselves.

        """
        self._n_hash_tables = n_hash_tables
        self._n_projections = n_projections
        self._n_hash = n_hash_tables * n_projections
        self._index_only = index_only

    @property
    def n_hash_tables(self):
        return self._n_hash_tables

    @property
    def n_projections(self):
        return self._n_projections

    @property
    def index_only(self):
        return self._index_only

    def _generate_random_projections(self, n_hash: int, d: int) -> np.ndarray:
        """
        Sample n_hash random unit vectors from a d-dimensional sphere.

        Args:
            n_hash: Number of vectors to sample
            d: Dimensionality of each vector
        Returns:
            np.ndarray: n_hash x d matrix of random unit vectors
        """
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

        if self._index_only:
            return RandomProjectionLSHModel(
                self._n_hash_tables, self._n_projections, index, P
            )
        else:
            return RandomProjectionLSHModel(
                self._n_hash_tables, self._n_projections, index, P, X
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
            X: the input vectors if index_only is disabled
        """
        self._n_hash_tables = n_hash_tables
        self._n_projections = n_projections
        self._index = index
        self._P = P
        self._X = X

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
            list[list[int]]: list of candidate neighbors for each query
            or
            list[list[np.ndarray]]: list of candidate neighbors for each query if index_only is disabled
        """
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

            if self._X is not None:
                all_candidates.append([self._X[i] for i in q_candidates])
            else:
                all_candidates.append(list(q_candidates))

        return all_candidates

    def save(self, save_dir: str):
        """Write the model to save dir."""
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

    @classmethod
    def load(cls, save_dir: str):
        """Read the model from save dir."""
        npz_path = os.path.join(save_dir, "arrays.npz")
        with np.load(npz_path) as data:
            P = data["P"]
            X = data.get("X")

        attr_path = os.path.join(save_dir, "attributes.pkl")
        with open(attr_path, "wb") as f:
            attrs = pickle.load(f)

        return cls(
            n_hash_tables=attrs["n_hash_tables"],
            n_projections=attrs["n_projections"],
            index=attrs["index"],
            P=P,
            X=X,
        )
