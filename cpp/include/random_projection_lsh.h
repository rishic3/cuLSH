#ifndef RANDOM_PROJECTION_LSH_H
#define RANDOM_PROJECTION_LSH_H

#include <Eigen/Dense>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

using namespace Eigen;

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
     * @param index_only If enabled, only store the LSH index and not the input vectors. The
     * subsequent LSH model will return the vector indices in the original dataset rather than the
     * vectors themselves.
     */
    RandomProjectionLSH(int n_hash_tables, int n_projections, bool index_only = true);

    int get_n_hash_tables() const { return n_hash_tables_; }
    int get_n_projections() const { return n_projections_; }
    int get_index_only() const { return index_only_; }

    /**
     * @brief Fit the RandomProjectionLSH model.
     * @param X The input vectors.
     * @return The fitted model.
     */
    std::unique_ptr<RandomProjectionLSHModel> fit(const MatrixXd& X);

private:
    int n_hash_tables_;
    int n_projections_;
    int n_hash_;
    bool index_only_;

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
 * @brief Hash a vector of integers.
 * @param vec The vector of integers to hash.
 * @return The hash of the vector.
 */
struct VectorHasher {
    std::size_t operator()(const std::vector<int>& vec) const;
};

/**
 * @class RandomProjectionLSHModel
 * @brief Model produced by RandomProjectionLSH.fit()
 */
class RandomProjectionLSHModel {
public:
    /**
     * @brief Initialize the RandomProjectionLSHModel.
     * @param n_hash_tables The number of hash tables.
     * @param n_projections The number of random hyperplanes (hash functions) per hash table.
     * @param index The index of the input vectors.
     * @param P The n_hash x d matrix of normal unit vectors.
     * @param X The input vectors if index_only is disabled.
     */
    RandomProjectionLSHModel(
        int n_hash_tables, int n_projections,
        std::vector<std::unordered_map<std::vector<int>, std::vector<int>, VectorHasher>> index, MatrixXd P,
        std::optional<MatrixXd> X = std::nullopt);

    int get_n_hash_tables() const { return n_hash_tables_; }
    int get_n_projections() const { return n_projections_; }
    int get_index_only() const { return X_.has_value(); }

    /**
     * @brief Query the RandomProjectionLSHModel.
     * @param Q The m x d matrix of query vectors.
     * @return The m x n_hash matrix of signature bit vectors.
     */
    MatrixXi query(const MatrixXd& Q);
    /**
     * @brief Save the RandomProjectionLSHModel to a directory.
     * @param save_dir The directory to save the model.
     */
    void save(const std::string& save_dir);
    /**
     * @brief Load the RandomProjectionLSHModel from a directory.
     * @param save_dir The directory to load the model.
     */
    static std::unique_ptr<RandomProjectionLSHModel> load(const std::string& save_dir);

private:
    int n_hash_tables_;
    int n_projections_;
    std::vector<std::unordered_map<std::vector<int>, std::vector<int>, VectorHasher>> index_;
    MatrixXd P_;
    std::optional<MatrixXd> X_;

    /**
     * @brief Hash the query matrix Q using the matrix of normal unit vectors P.
     * @param Q The m x d matrix of query vectors.
     * @param P The n_hash x d matrix of normal unit vectors.
     * @return The m x n_hash matrix of signature bit vectors.
     */
    MatrixXi hash(const MatrixXd& Q, const MatrixXd& P);
};

#endif