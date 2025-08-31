#include <random_projection_lsh.h>

#include <Eigen/Dense>
#include <memory>
#include <optional>
#include <random>
#include <unordered_map>
#include <vector>

using namespace Eigen;

std::size_t VectorHasher::operator()(const VectorXi& vec) const {
    // https://stackoverflow.com/questions/20511347/a-good-hash-function-for-a-vector/72073933#72073933
    std::size_t seed = vec.size();
    for (int x : vec) {
        x = ((x >> 16) ^ x) * 0x45d9f3b;
        x = ((x >> 16) ^ x) * 0x45d9f3b;
        x = (x >> 16) ^ x;
        seed ^= x + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
}

RandomProjectionLSH::RandomProjectionLSH(int n_hash_tables, int n_projections, bool index_only,
                                         unsigned int seed)
    : n_hash_tables(n_hash_tables), n_projections(n_projections),
      n_hash(n_hash_tables * n_projections), index_only(index_only), seed(seed), rng(seed),
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

std::unique_ptr<RandomProjectionLSHModel> RandomProjectionLSH::fit(const MatrixXd& X) {
    int d = X.cols();
    MatrixXd P = generate_random_projections(n_hash, d);
    MatrixXi H_x = hash(X, P);

    // index is a vector of maps for each hash table
    // each map stores (hash table signature -> vector of indices in X)
    std::vector<std::unordered_map<std::vector<int>, std::vector<int>, VectorHasher>> index(
        n_hash_tables);

    for (int i = 0; i < n_hash; ++i) {
        VectorXi signature = H_x.row(i);

        for (int j = 0; j < n_hash_tables; ++j) {
            int table_start = j * n_projections;
            VectorXi table_signature = signature(seqN(table_start, n_projections));

            std::unordered_map<std::vector<int>, std::vector<int>, VectorHasher>& table_map = index[j];
            if (table_map.find(table_signature) == table_map.end()) {
                table_map.insert({table_signature, {i}});
            } else {
                table_map.at(table_signature).push_back(i);
            }
        }
    }

    if (index_only) {
        return std::make_unique<RandomProjectionLSHModel>(n_hash_tables, n_projections, index, P);
    } else {
        return std::make_unique<RandomProjectionLSHModel>(n_hash_tables, n_projections, index, P, X);
    }
}
