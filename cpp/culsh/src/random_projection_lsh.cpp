#include <random_projection_lsh.h>

#include <Eigen/Dense>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace Eigen;
using namespace std;
namespace fs = filesystem;

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
    MatrixXd::Index d = X.cols();
    MatrixXd P = generate_random_projections(n_hash, d);
    MatrixXi H_x = hash(X, P);

    // index is a vector of maps for each hash table
    // each map stores (hash table signature -> vector of indices in X)
    vector<unordered_map<VectorXi, vector<int>, VectorHasher>> index(n_hash_tables);

    for (MatrixXd::Index i = 0; i < H_x.rows(); ++i) {
        VectorXi signature = H_x.row(i);

        for (int j = 0; j < n_hash_tables; ++j) {
            int table_start = j * n_projections;
            VectorXi table_signature = signature(seqN(table_start, n_projections));

            auto& hash_table = index[j];
            auto it = hash_table.find(table_signature);
            if (it == hash_table.end()) {
                hash_table.insert({table_signature, {i}});
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
    : n_hash_tables(n_hash_tables), n_projections(n_projections), index(std::move(index)),
      P(std::move(P)), X(std::move(X)) {}

MatrixXi RandomProjectionLSHModel::hash(const MatrixXd& Q, const MatrixXd& P) {
    MatrixXd prod = Q * P.transpose();
    return prod.array().sign().cast<int>();
}

template <typename ResultType>
vector<vector<ResultType>> RandomProjectionLSHModel::query_impl(const MatrixXd& Q) {
    MatrixXi H_q = hash(Q, P);
    vector<vector<ResultType>> all_candidates(Q.rows());

    for (MatrixXd::Index i = 0; i < H_q.rows(); ++i) {
        VectorXi q_signature = H_q.row(i);
        unordered_set<int> q_candidates;

        // for each hash table, retrieve candidates that hashed to that table from index
        for (int j = 0; j < n_hash_tables; ++j) {
            int table_start = j * n_projections;
            VectorXi q_table_signature = q_signature(seqN(table_start, n_projections));

            // get candidates from hash table j
            auto& hash_table = index[j];
            auto it = hash_table.find(q_table_signature);
            if (it != hash_table.end()) {
                const vector<int>& table_candidates = it->second;
                for (int candidate : table_candidates) {
                    q_candidates.insert(candidate);
                }
            }
        }

        all_candidates.reserve(q_candidates.size());
        if constexpr (is_same_v<ResultType, int>) {
            // store vector of indices
            all_candidates[i].assign(q_candidates.begin(), q_candidates.end());
        } else { // ResultType == VectorXd
            // store vector of vectors
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
        throw runtime_error("Input data X was not stored in the model via store_data=True. Use "
                            "query_indices() instead.");
    }
    return query_impl<VectorXd>(Q);
}

void RandomProjectionLSHModel::save_matrix_binary(const MatrixXd& mat, const fs::path& file_path) {
    ofstream file(file_path, ios::binary);
    MatrixXd::Index n_rows = mat.rows(), n_cols = mat.cols();

    // write dimensions
    file.write(reinterpret_cast<const char*>(&n_rows), sizeof(n_rows));
    file.write(reinterpret_cast<const char*>(&n_cols), sizeof(n_cols));

    // write data
    file.write(reinterpret_cast<const char*>(mat.data()),
               n_rows * n_cols * sizeof(MatrixXd::Scalar));
}

MatrixXd RandomProjectionLSHModel::load_matrix_binary(const fs::path& file_path) {
    ifstream file(file_path, ios::binary);
    MatrixXd::Index n_rows = 0, n_cols = 0;

    // read dimensions
    file.read(reinterpret_cast<char*>(&n_rows), sizeof(n_rows));
    file.read(reinterpret_cast<char*>(&n_cols), sizeof(n_cols));

    // read data
    MatrixXd mat(n_rows, n_cols);
    file.read(reinterpret_cast<char*>(mat.data()), n_rows * n_cols * sizeof(MatrixXd::Scalar));
    return mat;
}

void RandomProjectionLSHModel::save_index_binary(
    const vector<unordered_map<VectorXi, vector<int>, VectorHasher>>& index,
    const fs::path& file_path) {
    ofstream file(file_path, ios::binary);
    ssize_t n_hash_tables = index.size();
    file.write(reinterpret_cast<const char*>(&n_hash_tables), sizeof(n_hash_tables));

    for (const auto& hash_table : index) {
        ssize_t table_size = hash_table.size();
        file.write(reinterpret_cast<const char*>(&table_size), sizeof(table_size));

        for (const auto& [signature, candidate_indices] : hash_table) {
            // write signature
            VectorXi::Index n_cols = signature.cols();
            file.write(reinterpret_cast<const char*>(&n_cols), sizeof(n_cols));
            file.write(reinterpret_cast<const char*>(signature.data()),
                       n_cols * sizeof(VectorXi::Scalar));

            // write indices
            ssize_t n_candidates = candidate_indices.size();
            file.write(reinterpret_cast<const char*>(&n_candidates), sizeof(n_candidates));
            file.write(reinterpret_cast<const char*>(candidate_indices.data()),
                       n_candidates * sizeof(int));
        }
    }
}

vector<unordered_map<VectorXi, vector<int>, VectorHasher>>
RandomProjectionLSHModel::load_index_binary(const fs::path& file_path) {
    ifstream file(file_path, ios::binary);
    ssize_t n_hash_tables = 0;
    file.read(reinterpret_cast<char*>(&n_hash_tables), sizeof(n_hash_tables));

    vector<unordered_map<VectorXi, vector<int>, VectorHasher>> index(n_hash_tables);

    for (ssize_t i = 0; i < n_hash_tables; ++i) {
        ssize_t table_size = 0;
        file.read(reinterpret_cast<char*>(&table_size), sizeof(table_size));

        for (ssize_t j = 0; j < table_size; ++j) {
            // read signature
            VectorXi::Index n_cols = 0;
            file.read(reinterpret_cast<char*>(&n_cols), sizeof(n_cols));
            VectorXi signature(n_cols);
            file.read(reinterpret_cast<char*>(signature.data()), n_cols * sizeof(VectorXi::Scalar));

            // read indices
            ssize_t n_candidates = 0;
            file.read(reinterpret_cast<char*>(&n_candidates), sizeof(n_candidates));
            vector<int> candidate_indices(n_candidates);
            file.read(reinterpret_cast<char*>(candidate_indices.data()),
                      n_candidates * sizeof(int));

            index[i][signature] = std::move(candidate_indices);
        }
    }
    return index;
}

void RandomProjectionLSHModel::save(const fs::path& save_dir) {
    if (!fs::exists(save_dir)) {
        fs::create_directory(save_dir);
    }

    // save matrix attributes
    save_matrix_binary(P, save_dir / "P.bin");
    if (X.has_value()) {
        save_matrix_binary(X.value(), save_dir / "X.bin");
    }

    // save index
}

unique_ptr<RandomProjectionLSHModel> RandomProjectionLSHModel::load(const fs::path& save_dir) {
    // todo
}
