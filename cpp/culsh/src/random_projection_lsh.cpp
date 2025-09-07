#include <random_projection_lsh.h>

#include <Eigen/Dense>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <random>
#include <unordered_map>
#include <vector>

using namespace Eigen;
using namespace std;
namespace fs = filesystem;

using IndexType = RandomProjectionLSHModel::IndexType;

vector<int8_t> to_compact_signature(const VectorXi& vec) {
    // compact representation of signature bit vector for faster hashing
    vector<int8_t> sig;
    sig.reserve(vec.size());
    for (int i = 0; i < vec.size(); ++i) {
        sig.push_back(vec(i) > 0 ? 1 : 0);
    }
    return sig;
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

RandomProjectionLSHModel RandomProjectionLSH::fit(const MatrixXd& X) {
    auto start_time = chrono::high_resolution_clock::now();

    MatrixXd::Index d = X.cols();
    MatrixXd P = generate_random_projections(n_hash, d);
    MatrixXi H_x = hash(X, P);

    // index is a vector of maps for each hash table
    // each map stores (hash table signature -> vector of indices in X)
    IndexType index(n_hash_tables);

    for (MatrixXd::Index i = 0; i < H_x.rows(); ++i) {
        VectorXi signature = H_x.row(i);

        for (int j = 0; j < n_hash_tables; ++j) {
            int table_start = j * n_projections;
            VectorXi table_signature = signature(seqN(table_start, n_projections));
            vector<int8_t> compact_sig = to_compact_signature(table_signature);

            index[j][compact_sig].push_back(i);
        }
    }

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> fit_time = end_time - start_time;
    clog << "Fit completed in " << fit_time.count() << " sec" << endl;

    if (store_data) {
        return RandomProjectionLSHModel(n_hash_tables, n_projections, X.rows(), index, P, X);
    } else {
        return RandomProjectionLSHModel(n_hash_tables, n_projections, X.rows(), index, P);
    }
}

// ================== RandomProjectionLSHModel ==================

RandomProjectionLSHModel::RandomProjectionLSHModel(int n_hash_tables, int n_projections,
                                                   size_t n_data_points, IndexType index,
                                                   MatrixXd P, optional<MatrixXd> X)
    : n_hash_tables(n_hash_tables), n_projections(n_projections), n_data_points(n_data_points),
      index(std::move(index)), P(std::move(P)), X(std::move(X)) {}

MatrixXi RandomProjectionLSHModel::hash(const MatrixXd& Q, const MatrixXd& P) {
    MatrixXd prod = Q * P.transpose();
    return prod.array().sign().cast<int>();
}

template <typename ResultType>
vector<vector<ResultType>> RandomProjectionLSHModel::query_impl(const MatrixXd& Q) {
    auto start_time = chrono::high_resolution_clock::now();

    MatrixXi H_q = hash(Q, P);
    vector<vector<ResultType>> all_candidates(Q.rows());

    // store a bitmask of seen candidates as an alternative to unordered_set. benefit is two-fold:
    // 1) avoid hash overhead of unordered_set.insert() per query, per hash table
    // 2) store q_candidates contiguously to make all_candidates.assign(...) iteration much faster
    vector<bool> q_seen;
    vector<int> q_candidates;
    q_seen.resize(n_data_points, false);

    for (MatrixXd::Index i = 0; i < H_q.rows(); ++i) {
        VectorXi q_signature = H_q.row(i);
        q_candidates.clear();

        for (int j = 0; j < n_hash_tables; ++j) {
            int table_start = j * n_projections;
            VectorXi q_table_signature = q_signature(seqN(table_start, n_projections));
            vector<int8_t> q_compact_table_signature = to_compact_signature(q_table_signature);

            auto& hash_table = index[j];
            auto it = hash_table.find(q_compact_table_signature);
            if (it != hash_table.end()) {
                const vector<int>& table_candidates = it->second;

                // check bitmask and add to candidates if not yet seen
                for (int candidate_idx : table_candidates) {
                    if (!q_seen[candidate_idx]) {
                        q_seen[candidate_idx] = true;
                        q_candidates.push_back(candidate_idx);
                    }
                }
            }
        }

        // reset bitmask
        for (int candidate_idx : q_candidates) {
            q_seen[candidate_idx] = false;
        }

        all_candidates[i].reserve(q_candidates.size());
        if constexpr (is_same_v<ResultType, int>) {
            all_candidates[i].assign(q_candidates.begin(), q_candidates.end());
        } else { // is_same_v<ResultType, VectorXd>
            for (int candidate_idx : q_candidates) {
                all_candidates[i].push_back(X.value().row(candidate_idx));
            }
        }
    }

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> query_time = end_time - start_time;
    clog << "Query completed in " << query_time.count() << " sec" << endl;

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
    if (!file.is_open()) {
        throw runtime_error("Could not open file '" + file_path.string() + "'");
    }

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
    if (!file.is_open()) {
        throw runtime_error("Could not open file '" + file_path.string() + "'");
    }

    MatrixXd::Index n_rows = 0, n_cols = 0;

    // read dimensions
    file.read(reinterpret_cast<char*>(&n_rows), sizeof(n_rows));
    file.read(reinterpret_cast<char*>(&n_cols), sizeof(n_cols));

    // read data
    MatrixXd mat(n_rows, n_cols);
    file.read(reinterpret_cast<char*>(mat.data()), n_rows * n_cols * sizeof(MatrixXd::Scalar));
    return mat;
}

void RandomProjectionLSHModel::save_index_binary(const IndexType& index,
                                                 const fs::path& file_path) {
    ofstream file(file_path, ios::binary);
    if (!file.is_open()) {
        throw runtime_error("Could not open file '" + file_path.string() + "'");
    }

    size_t n_hash_tables = index.size();
    file.write(reinterpret_cast<const char*>(&n_hash_tables), sizeof(n_hash_tables));

    for (const auto& hash_table : index) {
        size_t table_size = hash_table.size();
        file.write(reinterpret_cast<const char*>(&table_size), sizeof(table_size));

        for (const auto& [signature, candidate_indices] : hash_table) {
            // write signature
            size_t sig_size = signature.size();
            file.write(reinterpret_cast<const char*>(&sig_size), sizeof(sig_size));
            file.write(reinterpret_cast<const char*>(signature.data()), sig_size);

            // write indices
            size_t n_candidates = candidate_indices.size();
            file.write(reinterpret_cast<const char*>(&n_candidates), sizeof(n_candidates));
            file.write(reinterpret_cast<const char*>(candidate_indices.data()),
                       n_candidates * sizeof(int));
        }
    }
}

IndexType RandomProjectionLSHModel::load_index_binary(const fs::path& file_path) {
    ifstream file(file_path, ios::binary);
    if (!file.is_open()) {
        throw runtime_error("Could not open file '" + file_path.string() + "'");
    }

    size_t n_hash_tables = 0;
    file.read(reinterpret_cast<char*>(&n_hash_tables), sizeof(n_hash_tables));

    IndexType index(n_hash_tables);

    for (size_t i = 0; i < n_hash_tables; ++i) {
        size_t table_size = 0;
        file.read(reinterpret_cast<char*>(&table_size), sizeof(table_size));

        for (size_t j = 0; j < table_size; ++j) {
            // read signature
            size_t sig_size = 0;
            file.read(reinterpret_cast<char*>(&sig_size), sizeof(sig_size));
            vector<int8_t> signature(sig_size);
            file.read(reinterpret_cast<char*>(signature.data()), sig_size);

            // read indices
            size_t n_candidates = 0;
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
        fs::create_directories(save_dir);
    }

    // write matrix attributes
    save_matrix_binary(P, save_dir / "P.bin");
    if (X.has_value()) {
        save_matrix_binary(X.value(), save_dir / "X.bin");
    }

    // write index
    save_index_binary(index, save_dir / "index.bin");

    // write params
    fs::path attributes_path = save_dir / "attributes.txt";
    ofstream attributes(attributes_path);
    if (!attributes.is_open()) {
        throw runtime_error("Could not create file '" + attributes_path.string() + "'");
    }
    attributes << n_hash_tables << "\n" << n_projections << "\n" << n_data_points << endl;

    clog << "Model saved to " << save_dir << endl;
}

RandomProjectionLSHModel RandomProjectionLSHModel::load(const fs::path& save_dir) {
    if (!fs::exists(save_dir) || !fs::is_directory(save_dir)) {
        throw runtime_error("Directory '" + save_dir.string() + "' not found");
    }

    // read matrix attributes
    MatrixXd P = load_matrix_binary(save_dir / "P.bin");
    optional<MatrixXd> X;
    if (fs::exists(save_dir / "X.bin")) {
        X = load_matrix_binary(save_dir / "X.bin");
    }

    // read index
    IndexType index = load_index_binary(save_dir / "index.bin");

    // read params
    fs::path attributes_path = save_dir / "attributes.txt";
    ifstream attributes(attributes_path);
    if (!attributes.is_open()) {
        throw runtime_error("Could not open file '" + attributes_path.string() + "'");
    }

    int n_hash_tables, n_projections;
    size_t n_data_points;
    attributes >> n_hash_tables >> n_projections >> n_data_points;

    clog << "Model loaded from " << save_dir << endl;

    return RandomProjectionLSHModel(n_hash_tables, n_projections, n_data_points, index, P, X);
}
