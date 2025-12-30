import argparse
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from culsh.utils import compute_recall

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class LSHBenchmark(ABC):
    """Abstract base class for LSH benchmarking."""

    def __init__(self):
        self.args = None

    @property
    @abstractmethod
    def algorithm_name(self) -> str:
        """Name of the LSH algorithm."""
        pass

    @abstractmethod
    def add_algorithm_args(self, parser: argparse.ArgumentParser) -> None:
        """Add algorithm-specific arguments to the parser."""
        pass

    @abstractmethod
    def load_data(self, data_dir: Path) -> tuple[Any, Any]:
        """
        Load training and query data.

        Returns
        -------
        X : Training data
        Q : Query data
        """
        pass

    @abstractmethod
    def create_lsh(self) -> Any:
        """Create and return the LSH instance."""
        pass

    @abstractmethod
    def get_ground_truth_top_k(self, X_train, Q_test, query_idx: int, k: int) -> Any:
        """
        Get ground truth top-k indices for a query.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training data (numpy or cupy array)
        Q_test : array-like of shape (n_queries, n_features)
            Query data (numpy or cupy array)
        query_idx : int
            Index of the query in Q_test
        k : int
            Number of top neighbors to return

        Returns
        -------
        np.ndarray or cp.ndarray
            Indices of top-k ground truth neighbors
        """
        pass

    def _call_fit_query(self, lsh: Any, data: Any) -> Any:
        """Call fit_query with appropriate arguments"""
        assert self.args is not None
        if self.args.batch_size:
            return lsh.fit_query(data, batch_size=self.args.batch_size)
        return lsh.fit_query(data)

    def _call_query(self, model: Any, data: Any) -> Any:
        """Call query with appropriate arguments"""
        assert self.args is not None
        if self.args.batch_size:
            return model.query(data, batch_size=self.args.batch_size)
        return model.query(data)

    def add_common_args(self, parser: argparse.ArgumentParser) -> None:
        """Add common arguments shared by all LSH algorithms."""
        parser.add_argument(
            "-d",
            "--data-dir",
            type=str,
            required=True,
            help="Path to directory containing dataset",
        )
        parser.add_argument("-nt", "--n-hash-tables", type=int, default=16)
        parser.add_argument(
            "-nh", "--n-hashes", type=int, default=4, help="Number of hashes per table"
        )
        parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed")
        parser.add_argument(
            "-nq", "--n-queries", type=int, default=100, help="Number of queries"
        )
        parser.add_argument(
            "-ne",
            "--n-eval-queries",
            type=int,
            default=10,
            help="Number of evaluation queries",
        )
        parser.add_argument(
            "-k",
            "--recall-k",
            type=str,
            default="10,100",
            help="Comma-separated k values for recall@k",
        )
        parser.add_argument(
            "-r",
            "--results-dir",
            type=str,
            default=None,
            help="Path to directory to write results",
        )
        parser.add_argument(
            "-fq",
            "--fit-query",
            action="store_true",
            help="Use combined fit_query (n_queries will be used for fit)",
        )
        parser.add_argument(
            "-bs",
            "--batch-size",
            type=int,
            default=None,
            help="Query batch size to reduce peak memory usage",
        )
        parser.add_argument("-v", "--verbose", action="store_true")

    def parse_args(self) -> argparse.Namespace:
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(description=f"Benchmark {self.algorithm_name}")
        self.add_common_args(parser)
        self.add_algorithm_args(parser)
        return parser.parse_args()

    def evaluate_recall(self, X_train, Q_test, candidates, recall_k_values):
        """Evaluate recall@k for the first n_eval_queries queries."""
        assert self.args is not None

        n_eval = min(self.args.n_eval_queries, Q_test.shape[0], candidates.n_queries)

        indices = candidates.get_indices()
        offsets = candidates.get_offsets()

        candidate_counts = []
        queries_with_candidates = 0

        recall_at_k = {k: [] for k in recall_k_values}
        for i in tqdm(range(n_eval), desc="Evaluating recall", ncols=100):
            # Slice candidates for this query
            start, end = int(offsets[i]), int(offsets[i + 1])
            lsh_indices = indices[start:end]
            n_candidates = end - start
            candidate_counts.append(n_candidates)

            if n_candidates == 0:
                for k in recall_k_values:
                    recall_at_k[k].append(0.0)
                logger.debug(f"  Query {i}: no candidates")
                continue

            queries_with_candidates += 1

            # Compute recall@k for each k
            for k in recall_k_values:
                gt_indices = self.get_ground_truth_top_k(X_train, Q_test, i, k)
                recall_score = compute_recall(lsh_indices, gt_indices)
                recall_at_k[k].append(recall_score)

            if logger.isEnabledFor(logging.DEBUG):
                recalls_str = ", ".join(
                    f"R@{k}={recall_at_k[k][-1]:.4f}" for k in recall_k_values
                )
                logger.debug(f"  Query {i}: {recalls_str} (candidates={n_candidates})")

        # Compute averages
        avg_recall_at_k = {
            k: float(np.mean(recalls)) for k, recalls in recall_at_k.items()
        }
        mean_candidates = float(np.mean(candidate_counts))

        return {
            "n_eval_queries": n_eval,
            "queries_with_candidates": queries_with_candidates,
            "recall_at_k": avg_recall_at_k,
            "mean_candidates": mean_candidates,
            "per_query_recall_at_k": {
                k: [float(r) for r in recalls] for k, recalls in recall_at_k.items()
            },
        }

    def run(self) -> None:
        """Run the benchmark."""
        self.args = self.parse_args()

        if self.args.verbose:
            logger.setLevel(logging.DEBUG)

        recall_k_values = [int(k.strip()) for k in self.args.recall_k.split(",")]

        data_dir = Path(self.args.data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"Data dir '{data_dir}' not found")

        X, Q = self.load_data(data_dir)

        logger.info(
            f"Parameters: n_hash_tables={self.args.n_hash_tables}, "
            f"n_hashes={self.args.n_hashes}, seed={self.args.seed}"
        )

        lsh = self.create_lsh()

        if self.args.fit_query:
            X_train = X[: self.args.n_queries]
            Q_test = X_train

            batch_msg = (
                f" (batch_size={self.args.batch_size})" if self.args.batch_size else ""
            )
            logger.info(
                f"Running fit_query() on {X_train.shape[0]} samples{batch_msg}..."
            )
            start_time = time.time()
            candidates = self._call_fit_query(lsh, X_train)
            fit_query_time = time.time() - start_time
            logger.info(f"fit_query() completed in {fit_query_time:.2f}s")

            fit_time = fit_query_time
            query_time = -1
        else:
            X_train = X
            logger.info(f"Running fit() on {X_train.shape[0]} samples...")
            start_time = time.time()
            model = lsh.fit(X_train)
            fit_time = time.time() - start_time
            logger.info(f"fit() completed in {fit_time:.2f}s")

            logger.debug(f"Index size: {model.index.size_bytes() / 1024**3:.2f} GB")

            Q_test = Q[: self.args.n_queries]
            batch_msg = (
                f" (batch_size={self.args.batch_size})" if self.args.batch_size else ""
            )
            logger.info(f"Running query() on {Q_test.shape[0]} queries{batch_msg}...")
            start_time = time.time()
            candidates = self._call_query(model, Q_test)
            query_time = time.time() - start_time
            logger.info(f"query() completed in {query_time:.2f}s")

        logger.info(
            f"Total candidates: {candidates.n_total_candidates} "
            f"({candidates.n_total_candidates * 4 / 1024**3:.2f} GB)"
        )

        # Evaluate recall
        n_queries = Q_test.shape[0]
        if candidates.n_queries != n_queries:
            logger.warning(
                "Candidates length does not match number of queries "
                f"(candidates.n_queries={candidates.n_queries}, n_queries={n_queries})."
            )

        n_eval = min(self.args.n_eval_queries, n_queries, candidates.n_queries)
        if n_eval == 0:
            logger.warning("No queries to evaluate (no candidates returned).")
            return

        recall_results = self.evaluate_recall(
            X_train, Q_test, candidates, recall_k_values
        )

        # Save or print results
        self._output_results(
            fit_time, query_time, recall_results, recall_k_values, n_eval
        )

    def _output_results(
        self, fit_time, query_time, recall_results, recall_k_values, n_eval
    ):
        """Output results to file or console."""
        assert self.args is not None

        if self.args.results_dir:
            report_path = (
                f"{self.args.results_dir}/report_{self.algorithm_name.lower()}_"
                f"h{self.args.n_hash_tables}_p{self.args.n_hashes}_"
                f"{time.strftime('%Y%m%d_%H%M%S')}.json"
            )
            os.makedirs(self.args.results_dir, exist_ok=True)

            with open(report_path, "w") as f:
                runtimes = (
                    {"fit_query_time": fit_time}
                    if self.args.fit_query
                    else {"fit_time": fit_time, "query_time": query_time}
                )
                report_data = {
                    "algorithm": self.algorithm_name,
                    "params": {
                        "n_hash_tables": self.args.n_hash_tables,
                        "n_hashes": self.args.n_hashes,
                        "seed": self.args.seed,
                        "n_queries": self.args.n_queries,
                        "recall_k": recall_k_values,
                        "mode": "fit_query" if self.args.fit_query else "fit+query",
                    },
                    "runtimes": runtimes,
                    "recall_evaluation": recall_results,
                }
                json.dump(report_data, f, indent=4)

            logger.info(f"Report saved to {report_path}")
        else:
            logger.info("=" * 50)
            logger.info("Results:")
            if self.args.fit_query:
                logger.info(f"  fit_query_time: {fit_time:.4f}s")
            else:
                logger.info(f"  fit_time: {fit_time:.4f}s")
                logger.info(f"  query_time: {query_time:.4f}s")
            logger.info(
                f"  queries with candidates: {recall_results['queries_with_candidates']}/{n_eval}"
            )
            logger.info(f"  mean candidates: {recall_results['mean_candidates']:.1f}")
            for k, recall in recall_results["recall_at_k"].items():
                logger.info(f"  recall@{k}: {recall:.4f}")
            logger.info("=" * 50)
