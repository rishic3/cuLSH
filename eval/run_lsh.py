import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.append("../python")

from random_projection_lsh import RandomProjectionLSH


def read_fvecs(fp):
    a = np.fromfile(fp, dtype="int32")
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy().view("float32")


def get_gt_top_k_indices(q, X, k=1000):
    q_norm = q / np.linalg.norm(q)
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    cos_sims = np.dot(X_norm, q_norm)

    # reverse - higher similarity is better
    return np.argsort(cos_sims)[-k:][::-1]


def recall(lsh_indices, gt_top_k_indices):
    lsh_set = set(lsh_indices)
    gt_set = set(gt_top_k_indices)

    intersection = lsh_set.intersection(gt_set)
    recall_score = len(intersection) / len(gt_set)

    return recall_score, len(intersection), len(gt_set)


def main():
    parser = argparse.ArgumentParser(
        description="Test RandomProjection LSH on SIFT dataset"
    )
    parser.add_argument("-nh", "--n-hash-tables", type=int, default=16)
    parser.add_argument("-np", "--n-projections", type=int, default=4)
    parser.add_argument("-s", "--seed", type=int, default=None)
    parser.add_argument("-d", "--data-dir", type=str, default="data/sift")
    parser.add_argument("-nq", "--num-queries", type=int, default=100)
    parser.add_argument("-s", "--save-model", type=str, default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data dir '{data_dir}' not found. download_sift1m first"
        )

    X = read_fvecs(data_dir / "sift_base.fvecs")
    Q = read_fvecs(data_dir / "sift_query.fvecs")

    print(f"Data shape: {X.shape}")
    print(f"Query shape: {Q.shape}")

    Q_test = Q[: args.num_queries]
    print(f"Using {len(Q_test)} test queries")

    lsh = RandomProjectionLSH(
        n_hash_tables=args.n_hash_tables,
        n_projections=args.n_projections,
        seed=args.seed,
    )

    print("Created LSH Model:")
    print(f"  n_hash_tables: {lsh.n_hash_tables}")
    print(f"  n_projections: {lsh.n_projections}")
    print(f"  seed: {lsh.seed}")

    print("running fit()...")
    start_time = time.time()
    model = lsh.fit(X)
    fit_time = time.time() - start_time
    print(f"fit() completed in {fit_time:.2f}s")

    # Query model
    print("running query()...")
    start_time = time.time()
    all_neighbors = model.query(Q_test)
    query_time = time.time() - start_time
    print(f"query() completed in {query_time:.2f}s")

    first_query = Q_test[0]
    lsh_indices = all_neighbors[0]

    if len(lsh_indices) > 0:
        k = len(lsh_indices)
        top_k_indices = get_gt_top_k_indices(first_query, X, k)
        recall_score, intersection_size, gt_size = recall(lsh_indices, top_k_indices)
        print(f"First query recall: {recall_score:.4f} ({intersection_size}/{gt_size})")
    else:
        print("No candidates found for first query")

    # Save model if requested
    if args.save_model:
        print(f"Saving model to {args.save_model}...")
        model.save(args.save_model)
        print("Model saved successfully")


if __name__ == "__main__":
    main()
