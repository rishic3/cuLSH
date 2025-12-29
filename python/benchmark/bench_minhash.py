import argparse
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse
from bench_base import LSHBenchmark, logger

from culsh import MinHashLSH


class MinHashBenchmark(LSHBenchmark):
    """Benchmark for MinHash LSH."""

    @property
    def algorithm_name(self) -> str:
        return "MinHash"

    def add_algorithm_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--density",
            type=float,
            default=0.1,
            help="Density of random sparse matrix (if generating synthetic data)",
        )
        parser.add_argument(
            "-ns",
            "--n-samples",
            type=int,
            default=10000,
            help="Number of samples for synthetic data",
        )
        parser.add_argument(
            "-nf",
            "--n-features",
            type=int,
            default=1000,
            help="Number of features for synthetic data",
        )
        parser.add_argument(
            "-qs",
            "--query-split",
            type=float,
            default=0.01,
            help="Fraction of data to use as queries (for .dat files like Kosarak)",
        )
        parser.add_argument(
            "-ms",
            "--max-samples",
            type=int,
            default=None,
            help="Maximum number of samples to load (for large datasets)",
        )

    def load_data(self, data_dir: Path) -> tuple[Any, Any]:
        """
        Load sparse data from these formats (checked in order):
        1. base.npz + query.npz - Pre-split sparse matrices
        2. *.dat - Kosarak transaction format (space-separated item IDs per line)
        3. Synthetic - Generate random sparse data
        """
        assert self.args is not None

        # Check for pre-split npz files
        base_npz = data_dir / "base.npz"
        query_npz = data_dir / "query.npz"
        if base_npz.exists() and query_npz.exists():
            logger.info("Loading pre-split .npz files...")
            X = scipy.sparse.load_npz(base_npz)
            Q = scipy.sparse.load_npz(query_npz)
            return X, Q

        # Check for .dat file
        dat_files = list(data_dir.glob("*.dat"))
        if dat_files:
            return self._load_transaction_file(dat_files[0])

        # Fall back to synthetic data
        return self._generate_synthetic_data()

    def _load_transaction_file(self, filepath: Path) -> tuple[Any, Any]:
        """
        Load transaction data from a .dat file.
        Format: each line is a space-separated list of item IDs (integers).
        Each line represents a set/transaction.
        """
        assert self.args is not None
        logger.info(f"Loading transaction data from {filepath}...")

        # Read transactions
        transactions = []
        max_item_id = 0
        max_samples = self.args.max_samples

        with open(filepath, "r") as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                items = [int(x) for x in line.strip().split() if x]
                if items:
                    transactions.append(items)
                    max_item_id = max(max_item_id, max(items))

        n_samples = len(transactions)
        n_features = max_item_id + 1  # Item IDs are 0-indexed after we shift

        logger.info(f"Loaded {n_samples} transactions with {n_features} unique items")

        # Build CSR matrix
        # Shift item IDs to be 0-indexed if needed
        min_item_id = min(min(t) for t in transactions)
        if min_item_id > 0:
            transactions = [[x - min_item_id for x in t] for t in transactions]
            n_features = max_item_id - min_item_id + 1

        row_indices = []
        col_indices = []
        for row_idx, items in enumerate(transactions):
            for item in items:
                row_indices.append(row_idx)
                col_indices.append(item)

        data = np.ones(len(row_indices), dtype=np.float32)
        X_full = scipy.sparse.csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(n_samples, n_features),
            dtype=np.float32,
        )

        # Split into train and query
        n_queries = max(1, int(n_samples * self.args.query_split))
        n_queries = min(n_queries, self.args.n_queries)

        # Use last n_queries as queries
        X = X_full[:-n_queries]
        Q = X_full[-n_queries:]

        logger.info(f"Split: {X.shape[0]} training, {Q.shape[0]} queries")

        return X, Q

    def _generate_synthetic_data(self) -> tuple[Any, Any]:
        """Generate synthetic random sparse data."""
        assert self.args is not None
        logger.info(
            f"Generating synthetic sparse data: "
            f"n_samples={self.args.n_samples}, n_features={self.args.n_features}, "
            f"density={self.args.density}"
        )
        X = scipy.sparse.random(
            self.args.n_samples,
            self.args.n_features,
            density=self.args.density,
            format="csr",
            dtype=np.float32,
        )
        X.data[:] = 1.0

        n_queries = min(self.args.n_queries, self.args.n_samples)
        Q = scipy.sparse.random(
            n_queries,
            self.args.n_features,
            density=self.args.density,
            format="csr",
            dtype=np.float32,
        )
        Q.data[:] = 1.0

        return X, Q

    def create_lsh(self) -> MinHashLSH:
        assert self.args is not None
        return MinHashLSH(
            n_hash_tables=self.args.n_hash_tables,
            n_hashes=self.args.n_hashes,
            seed=self.args.seed,
        )

    def _call_fit_query(self, lsh: Any, data: Any) -> Any:
        """MinHash doesn't support batch_size in fit_query."""
        return lsh.fit_query(data)

    def _call_query(self, model: Any, data: Any) -> Any:
        """MinHash doesn't support batch_size in query."""
        return model.query(data)

    def get_ground_truth_top_k(
        self, X_train, Q_test, query_idx: int, k: int
    ) -> np.ndarray:
        """Get top-k by Jaccard similarity."""
        q = Q_test[query_idx]

        # Compute Jaccard similarity between query and all training samples
        # Jaccard(A, B) = |A ∩ B| / |A ∪ B|
        jaccard_sims = np.zeros(X_train.shape[0])

        q_indices = set(q.indices)
        q_nnz = len(q_indices)

        for i in range(X_train.shape[0]):
            x_row = X_train.getrow(i)
            x_indices = set(x_row.indices)
            x_nnz = len(x_indices)

            intersection = len(q_indices & x_indices)
            union = q_nnz + x_nnz - intersection

            if union > 0:
                jaccard_sims[i] = intersection / union

        return np.argsort(jaccard_sims)[-k:][::-1]


if __name__ == "__main__":
    benchmark = MinHashBenchmark()
    benchmark.run()
