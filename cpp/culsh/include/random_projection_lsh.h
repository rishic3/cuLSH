#ifndef RANDOM_PROJECTION_LSH_H
#define RANDOM_PROJECTION_LSH_H

#include <Eigen/Dense>
#include <filesystem>
#include <optional>
#include <random>
#include <unordered_map>
#include <vector>

using namespace Eigen;
using namespace std;
namespace fs = filesystem;

class RandomProjectionLSHModel;

/**
 * @class RandomProjectionLSH
 * @brief Random projection LSH approximates cosine distance between vectors for ANN search.
 */
class RandomProjectionLSH {
public:
    /**
     * @brief Initialize the RandomProjectionLSH class.
     * @param n_hash_tables The number of hash tables. This parameter corresponds to an
     * OR-amplification of the locality-sensitive family. A higher value increases the probability
     * of finding a candidate neighbor. Corresponds to 'b' in the amplified probability (1 - (1 -
     * p^r)^b).
     * @param n_projections The number of random hyperplanes (hash functions) per hash table. This
     * parameter corresponds to an AND amplification of the locality-sensitive family. A higher
     * value decreases the probability of finding a candidate neighbor. Corresponds to 'r' in the
     * amplified probability (1 - (1 - p^r)^b).
     * @param store_data If enabled, store the input vectors in the resultant model. The
     * subsequent LSH model will return the vectors in the original dataset rather than just
     * the vector indices. Disabled by default.
     * @param seed Optional seed used to generate random projections, default is None.
     */
    RandomProjectionLSH(int n_hash_tables, int n_projections, bool store_data = false,
                        unsigned int seed = random_device{}());

    /**
     * @brief Initialize the RandomProjectionLSH class.
     * @param n_hash_tables The number of hash tables.
     * @param n_projections The number of random hyperplanes per hash table.
     * @param seed Custom seed for random projections.
     */
    RandomProjectionLSH(int n_hash_tables, int n_projections, unsigned int seed);

    int get_n_hash_tables() const { return n_hash_tables; }
    int get_n_projections() const { return n_projections; }
    int get_store_data() const { return store_data; }
    unsigned int get_seed() const { return seed; }

    /**
     * @brief Fit the RandomProjectionLSH model.
     * @param X The input vectors.
     * @return The fitted model.
     */
    RandomProjectionLSHModel fit(const MatrixXd& X);

private:
    int n_hash_tables;
    int n_projections;
    int n_hash;
    bool store_data;
    unsigned int seed;
    mt19937 rng;
    normal_distribution<double> normal_dist;

    /**
     * @brief Sample n_hash random unit vectors from a d-dimensional sphere.
     * @param n_hash The number of random unit vectors to generate.
     * @param d The dimensionality of the random unit vectors.
     * @return The n_hash x d matrix of random unit vectors.
     */
    MatrixXd generate_random_projections(int n_hash, int d);
    /**
     * @brief Hash the input vectors X using the matrix of normal unit vectors P.
     * @param X The n x d matrix of input vectors.
     * @param P The n_hash x d matrix of normal unit vectors.
     * @return The n x n_hash matrix of signature bit vectors.
     */
    MatrixXi hash(const MatrixXd& X, const MatrixXd& P);
};

/**
 * @brief Hash function for std::vector<int8_t>.
 * @param vec The vector of integers to hash.
 * @return The hash of the vector.
 */
struct VectorHasher {
    size_t operator()(const vector<int8_t>& vec) const {
        return hash<string_view>{}(
            string_view(reinterpret_cast<const char*>(vec.data()), vec.size()));
    }
};

/**
 * @class RandomProjectionLSHModel
 * @brief Model produced by RandomProjectionLSH.fit()
 */
class RandomProjectionLSHModel {
public:
    using IndexType = vector<unordered_map<vector<int8_t>, vector<int>, VectorHasher>>;

    /**
     * @brief Initialize the RandomProjectionLSHModel.
     * @param n_hash_tables The number of hash tables.
     * @param n_projections The number of random hyperplanes (hash functions) per hash table.
     * @param n_data_points The number of data points in the original dataset.
     * @param index The index of the input vectors.
     * @param P The n_hash x d matrix of normal unit vectors.
     * @param X The input vectors if store_data is disabled.
     */
    RandomProjectionLSHModel(int n_hash_tables, int n_projections, size_t n_data_points,
                             IndexType index, MatrixXd P, optional<MatrixXd> X = nullopt);

    int get_n_hash_tables() const { return n_hash_tables; }
    int get_n_projections() const { return n_projections; }
    int get_store_data() const { return X.has_value(); }

    /**
     * @brief Query the RandomProjectionLSHModel for candidate vector indices.
     * @param Q The m x d matrix of query vectors.
     * @return Vector of candidate neighbor indices for each query.
     */
    vector<vector<int>> query_indices(const MatrixXd& Q);

    /**
     * @brief Query the RandomProjectionLSHModel for candidate vectors.
     * @param Q The m x d matrix of query vectors.
     * @return Vector of candidate neighbor vectors for each query.
     * @throws runtime_error if X was not stored during model creation (store_data=False).
     */
    vector<vector<VectorXd>> query_vectors(const MatrixXd& Q);

    /**
     * @brief Save the RandomProjectionLSHModel to a directory.
     * @param save_dir The directory to save the model.
     */
    void save(const fs::path& save_dir);
    /**
     * @brief Load the RandomProjectionLSHModel from a directory.
     * @param save_dir The directory to load the model.
     */
    static RandomProjectionLSHModel load(const fs::path& save_dir);

private:
    int n_hash_tables;
    int n_projections;
    size_t n_data_points;
    IndexType index;
    MatrixXd P;
    optional<MatrixXd> X;

    /**
     * @brief Hash the query matrix Q using the matrix of normal unit vectors P.
     * @param Q The m x d matrix of query vectors.
     * @param P The n_hash x d matrix of normal unit vectors.
     * @return The m x n_hash matrix of signature bit vectors.
     */
    MatrixXi hash(const MatrixXd& Q, const MatrixXd& P);

    /**
     * @brief Template for query methods.
     * @tparam ResultType Either int for query_indices or VectorXd for query_vectors
     * @param Q The m x d matrix of query vectors.
     * @return Vector of candidate neighbors for each query.
     */
    template <typename ResultType> vector<vector<ResultType>> query_impl(const MatrixXd& Q);

    /**
     * @brief Helper method to save Eigen::MatrixXd in binary format.
     * @param mat Matrix to save.
     * @param file_path File path to save to.
     */
    static void save_matrix_binary(const MatrixXd& mat, const fs::path& file_path);

    /**
     * @brief Helper method to load Eigen::MatrixXd from binary format.
     * @param file_path File path to load from.
     */
    static MatrixXd load_matrix_binary(const fs::path& file_path);

    /**
     * @brief Helper method to save index in binary format.
     * @param index Index to save.
     * @param file_path File path to save to.
     */
    static void save_index_binary(const IndexType& index, const fs::path& file_path);

    /**
     * @brief Helper method to load index from binary format.
     * @param file_path File path to load from.
     * @throws runtime_error if invalid file_path is provided.
     */
    static IndexType load_index_binary(const fs::path& file_path);
};

#endif
