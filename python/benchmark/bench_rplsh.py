import argparse
from pathlib import Path
from typing import Any

import numpy as np
from bench_base import LSHBenchmark, logger
from faiss_wrapper import FaissLSHWrapper

from culsh import RPLSH


class RPLSHBenchmark(LSHBenchmark):
    """Benchmark for Random Projection LSH."""

    @property
    def algorithm_name(self) -> str:
        if self.args and self.args.cpu:
            return "RPLSH-CPU"
        return "RPLSH-GPU"

    def add_algorithm_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--cpu",
            action="store_true",
            help="Use FAISS (CPU) instead of cuLSH (GPU)",
        )
        parser.add_argument(
            "--n-candidates",
            type=int,
            default=1000,
            help="Number of candidates for FAISS to return (CPU mode only)",
        )
        parser.add_argument(
            "--dtype",
            type=str,
            default="float32",
            choices=["float32", "float64"],
            help="Data type of the input data",
        )

    def load_data(self, data_dir: Path) -> tuple[Any, Any]:
        """Load SIFT-style fvecs data."""
        assert self.args is not None
        X = self.read_fvecs(data_dir / "sift_base.fvecs", self.args.dtype)
        Q = self.read_fvecs(data_dir / "sift_query.fvecs", self.args.dtype)
        return X, Q

    def create_lsh(self) -> RPLSH | FaissLSHWrapper:
        assert self.args is not None
        if self.args.cpu:
            logger.info("Using FAISS IndexLSH (CPU)")
            return FaissLSHWrapper(
                n_hash_tables=self.args.n_hash_tables,
                n_hashes=self.args.n_hashes,
                n_candidates=self.args.n_candidates,
            )
        else:
            logger.info("Using cuLSH (GPU)")
            return RPLSH(
                n_hash_tables=self.args.n_hash_tables,
                n_hashes=self.args.n_hashes,
                seed=self.args.seed,
            )

    def get_ground_truth_top_k(self, X_train, Q_test, query_idx: int, k: int):
        """Get top-k by cosine similarity."""
        q = Q_test[query_idx]
        q_norm = q / np.linalg.norm(q)
        X_norm = X_train / np.linalg.norm(X_train, axis=1, keepdims=True)
        cos_sims = np.dot(X_norm, q_norm)
        return np.argsort(cos_sims)[-k:][::-1]

    @staticmethod
    def read_fvecs(fp, dtype: str):
        """Read fvecs file into numpy array."""
        a = np.fromfile(fp, dtype="int32")
        d = a[0]
        return a.reshape(-1, d + 1)[:, 1:].copy().view(dtype)


if __name__ == "__main__":
    benchmark = RPLSHBenchmark()
    benchmark.run()
