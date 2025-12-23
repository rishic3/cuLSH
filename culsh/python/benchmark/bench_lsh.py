import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np

from culsh import RPLSH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def read_fvecs(fp):
    """Read fvecs file into numpy array"""
    a = np.fromfile(fp, dtype="int32")
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy().view("float32")


def get_gt_top_k_indices(q, X, k):
    """Get top k indices of ground truth vectors by cosine similarity"""
    q_norm = q / np.linalg.norm(q)
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    cos_sims = np.dot(X_norm, q_norm)
    return np.argsort(cos_sims)[-k:][::-1]


def compute_recall(lsh_indices, gt_top_k_indices):
    """Calculate recall score"""
    if len(gt_top_k_indices) == 0:
        return 0.0
    lsh_set = set(lsh_indices)
    gt_set = set(gt_top_k_indices)
    intersection = lsh_set.intersection(gt_set)
    return len(intersection) / len(gt_set)


def candidates_to_list(candidates):
    """Convert Candidates object(s) to list of arrays.

    Handles both single Candidates object and list of Candidates (from batched queries).
    """
    if isinstance(candidates, list):
        # Batched results - concatenate all batches
        result = []
        for batch in candidates:
            indices = batch.get_indices()
            offsets = batch.get_offsets()
            for i in range(batch.n_queries):
                start, end = offsets[i], offsets[i + 1]
                result.append(indices[start:end])
        return result

    indices = candidates.get_indices()
    offsets = candidates.get_offsets()
    n_queries = candidates.n_queries

    result = []
    for i in range(n_queries):
        start, end = offsets[i], offsets[i + 1]
        result.append(indices[start:end])
    return result


def evaluate_recall(Q, X, all_neighbors, n_eval_queries):
    """Evaluate recall for the first n_eval_queries queries"""
    per_query_recall = []
    queries_with_candidates = 0

    for i in range(n_eval_queries):
        lsh_indices = all_neighbors[i]
        k = len(lsh_indices)

        if k == 0:
            per_query_recall.append(0.0)
            logger.debug(f"  Query {i}: no candidates")
            continue

        queries_with_candidates += 1
        gt_indices = get_gt_top_k_indices(Q[i], X, k)
        recall_score = compute_recall(lsh_indices, gt_indices)
        per_query_recall.append(recall_score)

        logger.debug(f"  Query {i}: recall={recall_score:.4f} (k={k})")

    avg_recall = np.mean(per_query_recall) if per_query_recall else 0.0

    return {
        "n_eval_queries": n_eval_queries,
        "queries_with_candidates": queries_with_candidates,
        "avg_recall": float(avg_recall),
        "per_query_recall": [float(r) for r in per_query_recall],
    }


def run_benchmark():
    parser = argparse.ArgumentParser(description="Benchmark cuLSH on SIFT dataset")
    parser.add_argument(
        "-d",
        "--data-dir",
        type=str,
        required=True,
        help="Path to directory containing SIFT dataset",
    )
    parser.add_argument("-nh", "--n-hash-tables", type=int, default=16)
    parser.add_argument(
        "-np", "--n-projections", type=int, default=4, help="Number of projections"
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
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data dir '{data_dir}' not found. Run download_sift1m first"
        )

    X = read_fvecs(data_dir / "sift_base.fvecs")
    Q = read_fvecs(data_dir / "sift_query.fvecs")

    logger.info(
        f"Parameters: n_hash_tables={args.n_hash_tables}, n_projections={args.n_projections}, seed={args.seed}"
    )

    lsh = RPLSH(
        n_hash_tables=args.n_hash_tables,
        n_projections=args.n_projections,
        seed=args.seed,
    )

    if args.fit_query:
        X_train = X[: args.n_queries]
        Q_test = X_train

        # Fit+Query
        logger.info(f"Running fit_query() on {len(X_train)} samples...")
        start_time = time.time()
        candidates = lsh.fit_query(X_train)
        fit_query_time = time.time() - start_time
        logger.info(f"fit_query() completed in {fit_query_time:.2f}s")

        fit_time = fit_query_time
        query_time = -1
    else:
        # Fit
        X_train = X
        logger.info(f"Running fit() on {len(X_train)} samples...")
        start_time = time.time()
        model = lsh.fit(X_train)
        fit_time = time.time() - start_time
        logger.info(f"fit() completed in {fit_time:.2f}s")

        logger.debug(f"Index size: {model.index.size_bytes() / 1024**3:.2f} GB")

        # Query
        Q_test = Q[: args.n_queries]
        batch_msg = f" (batch_size={args.batch_size})" if args.batch_size else ""
        logger.info(f"Running query() on {len(Q_test)} queries{batch_msg}...")
        start_time = time.time()
        candidates = model.query(Q_test, batch_size=args.batch_size)
        query_time = time.time() - start_time
        logger.info(f"query() completed in {query_time:.2f}s")

    # Handle batched results
    if isinstance(candidates, list):
        total_candidates = sum(c.n_total_candidates for c in candidates)
        logger.info(
            f"Total candidates: {total_candidates} ({total_candidates * 4 / 1024**3:.2f} GB) across {len(candidates)} batches"
        )
    else:
        logger.info(
            f"Total candidates: {candidates.n_total_candidates} ({candidates.n_total_candidates * 4 / 1024**3:.2f} GB)"
        )

    # Evaluate recall
    all_neighbors = candidates_to_list(candidates)
    if len(all_neighbors) != len(Q_test):
        n_candidates_queries = (
            sum(c.n_queries for c in candidates)
            if isinstance(candidates, list)
            else candidates.n_queries
        )
        logger.warning(
            "Candidates length does not match number of queries "
            f"(candidates.n_queries={n_candidates_queries}, len(all_neighbors)={len(all_neighbors)}, "
            f"len(Q_test)={len(Q_test)})."
        )
    n_eval = min(args.n_eval_queries, len(Q_test), len(all_neighbors))
    if n_eval == 0:
        logger.warning("No queries to evaluate (no candidates returned).")
        return

    recall_results = evaluate_recall(Q_test, X_train, all_neighbors, n_eval)

    # Save report
    if args.results_dir:
        report_path = f"{args.results_dir}/report_h{args.n_hash_tables}_p{args.n_projections}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs(args.results_dir, exist_ok=True)

        with open(report_path, "w") as f:
            runtimes = (
                {"fit_query_time": fit_time}
                if args.fit_query
                else {"fit_time": fit_time, "query_time": query_time}
            )
            report_data = {
                "params": {
                    "n_hash_tables": args.n_hash_tables,
                    "n_projections": args.n_projections,
                    "seed": args.seed,
                    "n_queries": args.n_queries,
                    "mode": "fit_query" if args.fit_query else "fit+query",
                },
                "runtimes": runtimes,
                "recall_evaluation": recall_results,
            }
            json.dump(report_data, f, indent=4)

        logger.info(f"Report saved to {report_path}")
    else:
        logger.info("=" * 50)
        logger.info("Results:")
        if args.fit_query:
            logger.info(f"  fit_query_time: {fit_time:.4f}s")
        else:
            logger.info(f"  fit_time: {fit_time:.4f}s")
            logger.info(f"  query_time: {query_time:.4f}s")
        logger.info(
            f"  average recall ({recall_results['queries_with_candidates']}/{n_eval} queries with candidates): {recall_results['avg_recall']:.4f}"
        )
        logger.info("=" * 50)


if __name__ == "__main__":
    run_benchmark()
