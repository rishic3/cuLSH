import os
import tempfile

import numpy as np
import pytest
import scipy.sparse
from datasketch import MinHash
from datasketch import MinHashLSH as DatasketchLSH

from culsh import MinHashLSH, MinHashLSHModel
from culsh.utils import compute_recall


def generate_sparse_data(
    n_samples: int, n_features: int, density: float, seed: int = 42
) -> scipy.sparse.csr_matrix:
    """Generate random sparse binary CSR matrix"""
    np.random.seed(seed)
    X = scipy.sparse.random(
        n_samples, n_features, density=density, format="csr", dtype=np.float32
    )
    X.data[:] = 1.0
    return X  # type: ignore[return-value]


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


def evaluate_recall_at_k(
    X: scipy.sparse.csr_matrix,
    Q: scipy.sparse.csr_matrix,
    indices: np.ndarray,
    offsets: np.ndarray,
    k: int,
) -> list[float]:
    """Evaluate recall@k for each query"""
    assert Q.shape is not None, "Shape is None"
    n_queries = Q.shape[0]
    recalls = []
    for q_idx in range(n_queries):
        start, end = int(offsets[q_idx]), int(offsets[q_idx + 1])
        lsh_indices = indices[start:end]
        gt_indices = get_jaccard_top_k(X, Q, q_idx, k)
        recalls.append(compute_recall(lsh_indices, gt_indices))
    return recalls


@pytest.mark.parametrize("density", [0.1, 0.5])
@pytest.mark.parametrize("n_samples", [500, 5000])
@pytest.mark.parametrize("n_hash_tables", [32])
@pytest.mark.parametrize("n_hashes", [8])
def test_minhash_recall_vs_datasketch(density, n_samples, n_hash_tables, n_hashes):
    """Test cuLSH MinHash against datasketch."""
    THRESHOLD = 0.03

    n_features = 100
    num_perm = n_hash_tables * n_hashes
    n_eval = 50
    k = 20
    seed = 42

    X = generate_sparse_data(n_samples, n_features, density)

    # --- Datasketch ---
    # Build MinHash signatures
    datasketch_minhashes = []
    for i in range(n_samples):
        mh = MinHash(num_perm=num_perm)
        for idx in X.getrow(i).indices:
            mh.update(str(idx).encode("utf-8"))
        datasketch_minhashes.append(mh)

    # Build LSH index
    ds_lsh = DatasketchLSH(num_perm=num_perm, params=(n_hash_tables, n_hashes))
    for i, mh in enumerate(datasketch_minhashes):
        ds_lsh.insert(str(i), mh)

    # Query and compute recall
    ds_recalls = []
    for q_idx in range(n_eval):
        ds_candidates = np.array(
            [int(str(x)) for x in ds_lsh.query(datasketch_minhashes[q_idx])]
        )
        gt = get_jaccard_top_k(X, X, q_idx, k)
        ds_recalls.append(compute_recall(ds_candidates, gt))

    ds_mean_recall = np.mean(ds_recalls)

    # --- cuLSH ---
    # Fit and query
    lsh = MinHashLSH(n_hash_tables=n_hash_tables, n_hashes=n_hashes, seed=seed)
    candidates = lsh.fit_query(X)

    indices = candidates.get_indices()
    offsets = candidates.get_offsets()

    # Compute recalls
    culsh_recalls = evaluate_recall_at_k(X, X, indices, offsets, k=k)
    culsh_mean_recall = np.mean(culsh_recalls)

    print(f"\nDatasketch recall@{k}: {ds_mean_recall:.4f}")
    print(f"cuLSH recall@{k}: {culsh_mean_recall:.4f}")

    diff = abs(ds_mean_recall - culsh_mean_recall)
    assert diff < THRESHOLD, f"Recall difference > {THRESHOLD}: {diff:.4f}"
    assert culsh_mean_recall <= 1.0  # sanity check


def test_minhash_save_load():
    """Test MinHashLSH save and load."""
    THRESHOLD = 0.00001

    n_hash_tables = 16
    n_hashes = 4
    n_samples = 500
    n_features = 100
    n_queries = 20
    seed = 42
    k = 20

    X = generate_sparse_data(n_samples, n_features, 0.1)
    Q = generate_sparse_data(n_queries, n_features, 0.1)

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
    def query_and_get_recall(
        model: MinHashLSHModel, X: scipy.sparse.csr_matrix, Q: scipy.sparse.csr_matrix
    ) -> float:
        candidates = model.query(Q)
        recalls = evaluate_recall_at_k(
            X, Q, candidates.get_indices(), candidates.get_offsets(), k
        )
        return float(np.mean(recalls))

    mean_recall = query_and_get_recall(model, X, Q)
    loaded_mean_recall = query_and_get_recall(loaded_model, X, Q)

    print(f"\nOriginal recall@{k}: {mean_recall:.4f}")
    print(f"Loaded recall@{k}: {loaded_mean_recall:.4f}")

    diff = abs(mean_recall - loaded_mean_recall)
    assert diff < THRESHOLD, f"Recall difference > {THRESHOLD}: {diff:.4f}"
