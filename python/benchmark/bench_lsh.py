import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from lsh import RandomProjectionLSH


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


def evaluate_recall(Q, X, all_neighbors, n_eval_queries, verbose=True):
    """Evaluate recall for the first n_eval_queries queries"""
    per_query_recall = []
    queries_with_candidates = 0

    for i in range(n_eval_queries):
        lsh_indices = all_neighbors[i]
        k = len(lsh_indices)

        if k == 0:
            per_query_recall.append(0.0)
            if verbose:
                print(f"  Query {i}: no candidates")
            continue

        queries_with_candidates += 1
        gt_indices = get_gt_top_k_indices(Q[i], X, k)
        recall_score = compute_recall(lsh_indices, gt_indices)
        per_query_recall.append(recall_score)

        if verbose:
            print(f"  Query {i}: recall={recall_score:.4f} (k={k})")

    avg_recall = np.mean(per_query_recall) if per_query_recall else 0.0

    return {
        "n_eval_queries": n_eval_queries,
        "queries_with_candidates": queries_with_candidates,
        "avg_recall": float(avg_recall),
        "per_query_recall": [float(r) for r in per_query_recall],
    }


def run_benchmark():
    parser = argparse.ArgumentParser(
        description="Benchmark pure-Python LSH on SIFT dataset"
    )
    parser.add_argument("-d", "--data-dir", type=str, required=True)
    parser.add_argument("-nh", "--n-hash-tables", type=int, default=16)
    parser.add_argument("-np", "--n-projections", type=int, default=4)
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-nq", "--n-queries", type=int, default=100)
    parser.add_argument("-ne", "--n-eval-queries", type=int, default=10)
    parser.add_argument("-r", "--results-dir", type=str, default=None)
    parser.add_argument("-o", "--save-dir", type=str, default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data dir '{data_dir}' not found. Run download_sift1m first"
        )

    X = read_fvecs(data_dir / "sift_base.fvecs")
    Q = read_fvecs(data_dir / "sift_query.fvecs")

    print(f"Data shape: {X.shape}")
    print(f"Query shape: {Q.shape}")

    Q_test = Q[: args.n_queries]
    print(f"Using {len(Q_test)} test queries")

    lsh = RandomProjectionLSH(
        n_hash_tables=args.n_hash_tables,
        n_projections=args.n_projections,
        seed=args.seed,
    )

    print()
    print("LSH Model:")
    print(f"  n_hash_tables: {lsh.n_hash_tables}")
    print(f"  n_projections: {lsh.n_projections}")
    print(f"  seed: {lsh.seed}")
    print()

    try:
        # Fit
        print("Running fit()...")
        start_time = time.time()
        model = lsh.fit(X)
        fit_time = time.time() - start_time
        print(f"fit() completed in {fit_time:.2f}s")
        print()

        # Query
        print("Running query()...")
        start_time = time.time()
        all_neighbors = model.query(Q_test)
        query_time = time.time() - start_time
        print(f"query() completed in {query_time:.2f}s")
        total_candidates = sum(len(n) for n in all_neighbors)
        print(f"Total candidates: {total_candidates}")
        print()

        # Evaluate recall
        n_eval = min(args.n_eval_queries, len(Q_test))
        print(f"Evaluating recall on first {n_eval} queries:")
        recall_results = evaluate_recall(Q_test, X, all_neighbors, n_eval, verbose=True)
        print()
        print(f"Average recall: {recall_results['avg_recall']:.4f}")
        print(f"Queries with candidates: {recall_results['queries_with_candidates']}/{n_eval}")
        print()

        # Save report
        if args.results_dir:
            report_path = f"{args.results_dir}/report_h{args.n_hash_tables}_p{args.n_projections}_{time.strftime('%Y%m%d_%H%M%S')}.json"
            os.makedirs(args.results_dir, exist_ok=True)

            with open(report_path, "w") as f:
                report_data = {
                    "params": {
                        "n_hash_tables": args.n_hash_tables,
                        "n_projections": args.n_projections,
                        "seed": args.seed,
                        "n_queries": args.n_queries,
                    },
                    "runtimes": {
                        "fit_time": fit_time,
                        "query_time": query_time,
                    },
                    "stats": {
                        "total_candidates": total_candidates,
                    },
                    "recall_evaluation": recall_results,
                }
                json.dump(report_data, f, indent=4)

            print(f"Report saved to {report_path}")
        else:
            print("Runtimes:")
            print(f"  fit_time: {fit_time:.2f}s")
            print(f"  query_time: {query_time:.2f}s")

        # Save model if requested
        if args.save_dir:
            print(f"Saving model to {args.save_dir}...")
            model.save(args.save_dir)
            print("Model saved successfully")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_benchmark()
