import os
import tempfile

import numpy as np
import pytest
from ref_lsh import RandomProjectionLSH as RefRPLSH
from utils import evaluate_recall_at_k, generate_dense_data

from culsh import RPLSH, RPLSHModel
from culsh.utils import compute_recall


def get_cosine_top_k(
    X: np.ndarray,
    Q: np.ndarray,
    query_idx: int,
    k: int,
) -> np.ndarray:
    """Get top-k indices from X by cosine similarity to Q[query_idx]"""
    q = Q[query_idx]
    q_norm = q / np.linalg.norm(q)
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    cos_sims = np.dot(X_norm, q_norm)
    return np.argsort(cos_sims)[-k:][::-1]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n_samples", [1000])
@pytest.mark.parametrize("n_hash_tables", [8, 32])
@pytest.mark.parametrize("n_hashes", [8, 16])
def test_rplsh_recall_vs_reference(dtype, n_samples, n_hash_tables, n_hashes):
    """Test cuLSH RPLSH against reference CPU implementation."""
    THRESHOLD = 0.05

    n_features = 100
    n_eval = 50
    k = 20
    seed = 42

    X = generate_dense_data(n_samples, n_features, dtype=dtype)

    # --- Reference CPU ---
    ref_lsh = RefRPLSH(n_hash_tables=n_hash_tables, n_projections=n_hashes, seed=seed)
    ref_model = ref_lsh.fit(X)

    # Query and compute recall
    ref_recalls = []
    ref_candidates = ref_model.query(X[:n_eval])
    for q_idx in range(n_eval):
        gt = get_cosine_top_k(X, X, q_idx, k)
        ref_recalls.append(compute_recall(np.array(ref_candidates[q_idx]), gt))

    ref_mean_recall = np.mean(ref_recalls)

    # --- cuLSH ---
    lsh = RPLSH(n_hash_tables=n_hash_tables, n_hashes=n_hashes, seed=seed)
    candidates = lsh.fit_query(X)

    indices = candidates.get_indices()
    offsets = candidates.get_offsets()

    # Compute recalls
    culsh_recalls = evaluate_recall_at_k(
        X, X, indices, offsets, get_top_k_fn=get_cosine_top_k, k=k
    )
    culsh_mean_recall = np.mean(culsh_recalls)

    print(f"\nReference recall@{k}: {ref_mean_recall:.4f}")
    print(f"cuLSH recall@{k}: {culsh_mean_recall:.4f}")

    diff = abs(ref_mean_recall - culsh_mean_recall)
    assert diff < THRESHOLD, f"Recall difference > {THRESHOLD}: {diff:.4f}"
    assert culsh_mean_recall <= 1.0  # sanity check


def test_rplsh_save_load():
    """Test RPLSH save and load."""
    THRESHOLD = 0.00001

    n_hash_tables = 16
    n_hashes = 8
    n_samples = 500
    n_features = 100
    n_queries = 20
    seed = 42
    k = 20

    X = generate_dense_data(n_samples, n_features)
    Q = generate_dense_data(n_queries, n_features, seed=123)

    # Fit model
    lsh = RPLSH(n_hash_tables=n_hash_tables, n_hashes=n_hashes, seed=seed)
    model = lsh.fit(X)

    # Save and reload
    with tempfile.TemporaryDirectory() as tempdir:
        model.save(os.path.join(tempdir, "test_rplsh.npz"))
        loaded_model = RPLSHModel.load(os.path.join(tempdir, "test_rplsh.npz"))

    # Check attributes
    for attr in [
        "n_hash_tables",
        "n_hashes",
        "n_features",
    ]:
        assert getattr(model, attr) == getattr(loaded_model, attr)

    for index_attr in [
        "n_total_candidates",
        "n_total_buckets",
        "n_hash_tables",
        "n_hashes",
        "sig_nbytes",
        "n_features",
        "seed",
    ]:
        assert getattr(model.index, index_attr) == getattr(
            loaded_model.index, index_attr
        )

    # Check recall
    def query_and_get_recall(model: RPLSHModel, X: np.ndarray, Q: np.ndarray) -> float:
        candidates = model.query(Q)
        recalls = evaluate_recall_at_k(
            X,
            Q,
            candidates.get_indices(),
            candidates.get_offsets(),
            get_top_k_fn=get_cosine_top_k,
            k=k,
        )
        return float(np.mean(recalls))

    mean_recall = query_and_get_recall(model, X, Q)
    loaded_mean_recall = query_and_get_recall(loaded_model, X, Q)

    print(f"\nOriginal recall@{k}: {mean_recall:.4f}")
    print(f"Loaded recall@{k}: {loaded_mean_recall:.4f}")

    diff = abs(mean_recall - loaded_mean_recall)
    assert diff < THRESHOLD, f"Recall difference > {THRESHOLD}: {diff:.4f}"
