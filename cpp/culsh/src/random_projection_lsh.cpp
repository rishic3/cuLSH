#include <random_projection_lsh.h>

#include <Eigen/Dense>
#include <memory>
#include <optional>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace Eigen;
using namespace std;

size_t VectorHasher::operator()(const VectorXi& vec) const {
    // from https://stackoverflow.com/questions/20511347/a-good-hash-function-for-a-vector/72073933#72073933
    size_t seed = vec.size();
    for (int x : vec) {
        x = ((x >> 16) ^ x) * 0x45d9f3b;
        x = ((x >> 16) ^ x) * 0x45d9f3b;
        x = (x >> 16) ^ x;
        seed ^= x + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
}

// ================== RandomProjectionLSH ==================

RandomProjectionLSH::RandomProjectionLSH(int n_hash_tables, int n_projections, bool store_data,
                                         unsigned int seed)
    : n_hash_tables(n_hash_tables), n_projections(n_projections),
      n_hash(n_hash_tables * n_projections), store_data(store_data), seed(seed), rng(seed),
      normal_dist(0.0, 1.0) {}

RandomProjectionLSH::RandomProjectionLSH(int n_hash_tables, int n_projections, unsigned int seed)
    : n_hash_tables(n_hash_tables), n_projections(n_projections),
      n_hash(n_hash_tables * n_projections), store_data(false), seed(seed), rng(seed),
      normal_dist(0.0, 1.0) {}

MatrixXd RandomProjectionLSH::generate_random_projections(int n_hash, int d) {
    MatrixXd random_vecs(n_hash, d);

    for (int i = 0; i < n_hash; ++i) {
        for (int j = 0; j < d; ++j) {
            random_vecs(i, j) = normal_dist(rng);
        }
    }

    VectorXd norms = random_vecs.rowwise().norm();
    random_vecs = random_vecs.cwiseQuotient(norms.replicate(1, d));

    return random_vecs;
}

MatrixXi RandomProjectionLSH::hash(const MatrixXd& X, const MatrixXd& P) {
    MatrixXd prod = X * P.transpose();
    return prod.array().sign().cast<int>();
}

unique_ptr<RandomProjectionLSHModel> RandomProjectionLSH::fit(const MatrixXd& X) {
    int d = X.cols();
    MatrixXd P = generate_random_projections(n_hash, d);
    MatrixXi H_x = hash(X, P);

    // index is a vector of maps for each hash table
    // each map stores (hash table signature -> vector of indices in X)
    vector<unordered_map<VectorXi, vector<int>, VectorHasher>> index(n_hash_tables);

    for (int i = 0; i < H_x.rows(); ++i) {
        VectorXi signature = H_x.row(i);

        for (int j = 0; j < n_hash_tables; ++j) {
            int table_start = j * n_projections;
            VectorXi table_signature = signature(seqN(table_start, n_projections));

            auto& table_map = index[j];
            auto it = table_map.find(table_signature);
            if (it == table_map.end()) {
                table_map.insert({table_signature, {i}});
            } else {
                it->second.push_back(i);
            }
        }
    }

    if (store_data) {
        return make_unique<RandomProjectionLSHModel>(n_hash_tables, n_projections, index, P, X);
    } else {
        return make_unique<RandomProjectionLSHModel>(n_hash_tables, n_projections, index, P);
    }
}

// ================== RandomProjectionLSHModel ==================

RandomProjectionLSHModel::RandomProjectionLSHModel(
    int n_hash_tables, int n_projections,
    vector<unordered_map<VectorXi, vector<int>, VectorHasher>> index, MatrixXd P,
    optional<MatrixXd> X)
    : n_hash_tables(n_hash_tables), n_projections(n_projections), index(move(index)),
      P(move(P)), X(move(X)) {}

MatrixXi RandomProjectionLSHModel::hash(const MatrixXd& Q, const MatrixXd& P) {
    MatrixXd prod = Q * P.transpose();
    return prod.array().sign().cast<int>();
}

template<typename ResultType>
vector<vector<ResultType>> RandomProjectionLSHModel::query_impl(const MatrixXd& Q) {
    MatrixXi H_q = hash(Q, P);
    vector<vector<ResultType>> all_candidates(Q.rows());

    for (int i = 0; i < H_q.rows(); ++i) {
        VectorXi q_signature = H_q.row(i);
        unordered_set<int> q_candidates;

        // for each hash table, retrieve candidates that hashed to that table from index
        for (int j = 0; j < n_hash_tables; ++j) {
            int table_start = j * n_projections;
            VectorXi q_table_signature = q_signature(seqN(table_start, n_projections));

            // get candidates from hash table j
            auto& table_map = index[j];
            auto it = table_map.find(q_table_signature);
            if (it != table_map.end()) {
                const vector<int>& table_candidates = it->second;
                for (int candidate : table_candidates) {
                    q_candidates.insert(candidate);
                }
            }
        }

        all_candidates.reserve(q_candidates.size());
        if constexpr (is_same_v<ResultType, int>) {
            // return vector of indices
            all_candidates[i].assign(q_candidates.begin(), q_candidates.end());
        } else {
            for (int candidate_idx : q_candidates) {
                all_candidates[i].push_back(X.value().row(candidate_idx));
            }
        }
    }
    return all_candidates;
}

vector<vector<int>> RandomProjectionLSHModel::query_indices(const MatrixXd& Q) {
    return query_impl<int>(Q);
}

vector<vector<VectorXd>> RandomProjectionLSHModel::query_vectors(const MatrixXd& Q) {
    if (!X.has_value()) {
        throw runtime_error("Input data X was not stored during model creation via store_data=True. Use query_indices().");
    }
    return query_impl<VectorXd>(Q);
}

void RandomProjectionLSHModel::save(const string& save_dir) {
    // todo
}

unique_ptr<RandomProjectionLSHModel> RandomProjectionLSHModel::load(const string& save_dir) {
    // todo
}
