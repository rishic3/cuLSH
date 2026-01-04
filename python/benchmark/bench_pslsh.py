import argparse
from pathlib import Path
from typing import Any

import numpy as np
from bench_base import LSHBenchmark, logger
from faiss_wrapper import FaissLSHWrapper

from culsh import PStableLSH


class PSLSHBenchmark(LSHBenchmark):
    """Benchmark for p-Stable LSH."""

    _ground_truth: np.ndarray | None = None

    @property
    def algorithm_name(self) -> str:
        if self.args and self.args.cpu:
            return "PSLSH-CPU"
        return "PSLSH-GPU"

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
        parser.add_argument(
            "--window-size",
            type=str,
            default="auto",
            help="Window size for p-stable LSH quantization ('auto' or integer)",
        )

    def load_data(self, data_dir: Path) -> tuple[Any, Any]:
        """Load SIFT-style fvecs data and ground truth."""
        assert self.args is not None
        X = self.read_fvecs(data_dir / "sift_base.fvecs", self.args.dtype)
        Q = self.read_fvecs(data_dir / "sift_query.fvecs", self.args.dtype)

        # Load ground truth (precomputed Euclidean nearest neighbors)
        gt_path = data_dir / "sift_groundtruth.ivecs"
        if gt_path.exists():
            self._ground_truth = self.read_ivecs(gt_path)
            logger.info(f"Loaded ground truth: {self._ground_truth.shape}")
        else:
            logger.warning(
                f"Ground truth file not found at {gt_path}, will compute directly"
            )
            self._ground_truth = None

        return X, Q

    def create_lsh(self) -> PStableLSH | FaissLSHWrapper:
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
            ws = self.args.window_size
            window_size: int | str = "auto" if ws == "auto" else int(ws)
            return PStableLSH(
                n_hash_tables=self.args.n_hash_tables,
                n_hashes=self.args.n_hashes,
                window_size=window_size,
                seed=self.args.seed,
            )

    def get_ground_truth_top_k(self, X_train, Q_test, query_idx: int, k: int):
        """Get top-k by Euclidean distance (from precomputed or computed)."""
        if self._ground_truth is not None:
            return self._ground_truth[query_idx, :k]
        else:
            q = Q_test[query_idx]
            dists = np.linalg.norm(X_train - q, axis=1)
            return np.argsort(dists)[:k]

    @staticmethod
    def read_fvecs(fp, dtype: str):
        """Read fvecs file into numpy array."""
        a = np.fromfile(fp, dtype="int32")
        d = a[0]
        return a.reshape(-1, d + 1)[:, 1:].copy().view(dtype)

    @staticmethod
    def read_ivecs(fp) -> np.ndarray:
        """Read ivecs file (ground truth indices) into numpy array."""
        a = np.fromfile(fp, dtype="int32")
        d = a[0]
        return a.reshape(-1, d + 1)[:, 1:].copy()


if __name__ == "__main__":
    benchmark = PSLSHBenchmark()
    benchmark.run()
