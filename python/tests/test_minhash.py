import os
import tempfile

import numpy as np
import pytest
import scipy.sparse
from ref_lsh import MinHashLSH as RefMinHashLSH
from utils import evaluate_recall_at_k, generate_sparse_data

from culsh import MinHashLSH, MinHashLSHModel
from culsh.utils import compute_recall


def get_jaccard_top_k(
    X: scipy.sparse.csr_matrix,
    Q: scipy.sparse.csr_matrix,
    query_idx: int,
    k: int,
) -> np.ndarray:
    """Get top-k indices from X by Jaccard similarity to Q[query_idx]"""
    q = Q.getrow(query_idx)
    # Intersection via dot product: X @ q.T
    intersections = X.dot(q.T).toarray().ravel()
    # Row-wise nnz counts
    x_nnz = np.diff(X.indptr)
    q_nnz = q.nnz
    # Union = |A| + |B| - |A ∩ B|
    unions = x_nnz + q_nnz - intersections
    # Jaccard = intersection / union
    jaccard_sims = np.where(unions > 0, intersections / unions, 0.0)
    return np.argsort(jaccard_sims)[-k:][::-1]


@pytest.mark.parametrize("density", [0.1, 0.5])
@pytest.mark.parametrize("n_samples", [1000])
@pytest.mark.parametrize("n_hash_tables", [8, 32, 64])
@pytest.mark.parametrize("n_hashes", [8, 16])
def test_minhash_recall_vs_reference(density, n_samples, n_hash_tables, n_hashes):
    """Test cuLSH MinHash against reference CPU implementation."""
    THRESHOLD = 0.05

    n_features = 100
    n_eval = 50
    k = 20
    seed = 42

    X = generate_sparse_data(n_samples, n_features, density)

    # --- Reference CPU ---
    ref_lsh = RefMinHashLSH(n_hash_tables=n_hash_tables, n_hashes=n_hashes, seed=seed)
    ref_model = ref_lsh.fit(X)

    # Query and compute recall
    ref_recalls = []
    ref_candidates = ref_model.query(X[:n_eval])
    for q_idx in range(n_eval):
        gt = get_jaccard_top_k(X, X, q_idx, k)
        ref_recalls.append(compute_recall(np.array(ref_candidates[q_idx]), gt))

    ref_mean_recall = np.mean(ref_recalls)

    # --- cuLSH ---
    lsh = MinHashLSH(n_hash_tables=n_hash_tables, n_hashes=n_hashes, seed=seed)
    candidates = lsh.fit_query(X)

    indices = candidates.get_indices()
    offsets = candidates.get_offsets()

    # Compute recalls
    culsh_recalls = evaluate_recall_at_k(
        X, X, indices, offsets, get_top_k_fn=get_jaccard_top_k, k=k  # type: ignore
    )
    culsh_mean_recall = np.mean(culsh_recalls)

    print(f"\nReference recall@{k}: {ref_mean_recall:.4f}")
    print(f"cuLSH recall@{k}: {culsh_mean_recall:.4f}")

    diff = abs(ref_mean_recall - culsh_mean_recall)
    assert diff < THRESHOLD, f"Recall difference > {THRESHOLD}: {diff:.4f}"
    assert culsh_mean_recall <= 1.0  # sanity check


@pytest.mark.parametrize("cupy", [False, True])
def test_minhash_save_load(cupy):
    """Test MinHashLSH save and load."""
    THRESHOLD = 0.00001

    n_hash_tables = 16
    n_hashes = 4
    n_samples = 500
    n_features = 100
    n_queries = 20
    seed = 42
    k = 20

    X = generate_sparse_data(n_samples, n_features, 0.1, cupy=cupy)
    Q = generate_sparse_data(n_queries, n_features, 0.1, cupy=cupy)

    # Fit model
    lsh = MinHashLSH(n_hash_tables=n_hash_tables, n_hashes=n_hashes, seed=seed)
    model = lsh.fit(X)

    # Save and reload
    with tempfile.TemporaryDirectory() as tempdir:
        model.save(os.path.join(tempdir, "test_minhash.npz"))
        loaded_model = MinHashLSHModel.load(os.path.join(tempdir, "test_minhash.npz"))

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
    def query_and_get_recall(model: MinHashLSHModel, X, Q) -> float:
        candidates = model.query(Q)
        recalls = evaluate_recall_at_k(
            X,
            Q,
            candidates.get_indices(),
            candidates.get_offsets(),
            get_top_k_fn=get_jaccard_top_k,  # type: ignore
            k=k,
        )
        return float(np.mean(recalls))

    mean_recall = query_and_get_recall(model, X, Q)
    loaded_mean_recall = query_and_get_recall(loaded_model, X, Q)

    print(f"\nOriginal recall@{k}: {mean_recall:.4f}")
    print(f"Loaded recall@{k}: {loaded_mean_recall:.4f}")

    diff = abs(mean_recall - loaded_mean_recall)
    assert diff < THRESHOLD, f"Recall difference > {THRESHOLD}: {diff:.4f}"


@pytest.mark.parametrize("cupy", [False, True])
def test_minhash_fit_query_consistency(cupy):
    """Verify fit() + query() gives same results as fit_query()"""
    n_samples = 500
    n_features = 100
    n_hash_tables = 16
    n_hashes = 8
    seed = 42

    X = generate_sparse_data(n_samples, n_features, density=0.2, cupy=cupy)

    lsh = MinHashLSH(n_hash_tables=n_hash_tables, n_hashes=n_hashes, seed=seed)

    # fit_query
    candidates1 = lsh.fit_query(X)

    # fit + query
    model = lsh.fit(X)
    candidates2 = model.query(X)

    np.testing.assert_array_equal(
        candidates1.get_indices(),
        candidates2.get_indices(),
        err_msg="Indices differ",
    )
    np.testing.assert_array_equal(
        candidates1.get_offsets(),
        candidates2.get_offsets(),
        err_msg="Offsets differ",
    )
