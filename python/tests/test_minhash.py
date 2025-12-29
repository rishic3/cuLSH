import numpy as np
import pytest
import scipy.sparse
from datasketch import MinHash
from datasketch import MinHashLSH as DatasketchLSH

from culsh import MinHashLSH


def generate_sparse_data(
    n_samples: int, n_features: int, density: float, seed: int = 42
) -> scipy.sparse.csr_matrix:
    """Generate random sparse binary CSR matrix."""
    np.random.seed(seed)
    X = scipy.sparse.random(
        n_samples, n_features, density=density, format="csr", dtype=np.float32
    )
    X.data[:] = 1.0
    return X  # type: ignore[return-value]


def compute_jaccard_top_k(X, query_idx: int, k: int) -> np.ndarray:
    """Compute ground truth top-k neighbors by Jaccard similarity."""
    q = X.getrow(query_idx)
    intersections = X.dot(q.T).toarray().ravel()
    x_nnz = np.diff(X.indptr)
    q_nnz = q.nnz
    unions = x_nnz + q_nnz - intersections
    jaccard_sims = np.where(unions > 0, intersections / unions, 0.0)
    return np.argsort(jaccard_sims)[-k:][::-1]


def compute_recall(candidates: set, ground_truth: np.ndarray) -> float:
    """Compute recall score."""
    if len(ground_truth) == 0:
        return 0.0
    gt_set = set(ground_truth)
    return len(candidates & gt_set) / len(gt_set)


@pytest.mark.parametrize("density", [0.1, 0.5])
@pytest.mark.parametrize("n_samples", [500, 5000])
@pytest.mark.parametrize("n_hash_tables", [32])
@pytest.mark.parametrize("n_hashes", [8])
def test_minhash_recall_vs_datasketch(density, n_samples, n_hash_tables, n_hashes):
    """Test cuLSH MinHash against datasketch."""
    THRESHOLD = 0.03

    n_features = 100
    num_perm = n_hash_tables * n_hashes
    n_queries = 50
    k = 20

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
    for q_idx in range(n_queries):
        ds_candidates = {int(str(x)) for x in ds_lsh.query(datasketch_minhashes[q_idx])}
        gt = compute_jaccard_top_k(X, q_idx, k)
        ds_recalls.append(compute_recall(ds_candidates, gt))

    ds_mean_recall = np.mean(ds_recalls)

    # --- cuLSH ---
    # Fit and query
    lsh = MinHashLSH(n_hash_tables=n_hash_tables, n_hashes=n_hashes, seed=42)
    candidates = lsh.fit_query(X)

    indices = candidates.get_indices()
    offsets = candidates.get_offsets()

    # Compute recalls
    culsh_recalls = []
    for q_idx in range(n_queries):
        start, end = offsets[q_idx], offsets[q_idx + 1]
        culsh_candidates = set(indices[start:end])
        gt = compute_jaccard_top_k(X, q_idx, k)
        culsh_recalls.append(compute_recall(culsh_candidates, gt))

    culsh_mean_recall = np.mean(culsh_recalls)

    print(f"\nDatasketch recall@{k}: {ds_mean_recall:.4f}")
    print(f"cuLSH recall@{k}: {culsh_mean_recall:.4f}")

    diff = abs(ds_mean_recall - culsh_mean_recall)
    assert diff < THRESHOLD, f"Recall difference > {THRESHOLD}: {diff:.4f}"
    assert culsh_mean_recall <= 1.0  # sanity check
