#include "bench_utils.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <stdexcept>

namespace bench {

float* read_fvecs(const std::string& filepath, int& n_vectors, int& dimensions) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file '" + filepath + "'");
    }

    // read input dimension
    int d;
    file.read(reinterpret_cast<char*>(&d), sizeof(int));

    file.seekg(0, std::ios::end);
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    int vector_size = sizeof(int) + d * sizeof(float);
    int n_vecs = file_size / vector_size;

    // allocate memory for data array (row-major: n_vectors x dimensions)
    float* data = new float[n_vecs * d];

    // read all vectors into float array
    for (int i = 0; i < n_vecs; ++i) {
        int dim;
        file.read(reinterpret_cast<char*>(&dim), sizeof(int));
        if (dim != d) {
            throw std::runtime_error("Inconsistent dimensions in file");
        }

        file.read(reinterpret_cast<char*>(&data[i * d]), d * sizeof(float));
    }

    n_vectors = n_vecs;
    dimensions = d;
    return data;
}

std::vector<int> get_gt_top_k_indices(const float* q, const float* X, int n_vectors, int dimensions,
                                      int k) {
    // normalize query vector
    float q_norm = 0.0f;
    for (int i = 0; i < dimensions; ++i) {
        q_norm += q[i] * q[i];
    }
    q_norm = std::sqrt(q_norm);

    std::vector<float> q_normalized(dimensions);
    for (int i = 0; i < dimensions; ++i) {
        q_normalized[i] = q[i] / q_norm;
    }

    std::vector<std::pair<float, int>> sim_idx;
    sim_idx.reserve(n_vectors);

    // compute cosine similarities
    for (int i = 0; i < n_vectors; ++i) {
        // normalize X[i] and compute dot product with normalized query
        float x_norm = 0.0f;
        for (int j = 0; j < dimensions; ++j) {
            float val = X[i * dimensions + j];
            x_norm += val * val;
        }
        x_norm = std::sqrt(x_norm);

        float cos_sim = 0.0f;
        for (int j = 0; j < dimensions; ++j) {
            cos_sim += (X[i * dimensions + j] / x_norm) * q_normalized[j];
        }

        sim_idx.push_back({cos_sim, i});
    }

    std::sort(sim_idx.begin(), sim_idx.end(), std::greater<std::pair<float, int>>());

    std::vector<int> top_k;
    for (int i = 0; i < std::min(k, static_cast<int>(sim_idx.size())); ++i) {
        top_k.push_back(sim_idx[i].second);
    }

    return top_k;
}

double calculate_recall(const std::vector<int>& lsh_indices, const std::vector<int>& gt_indices) {
    std::set<int> lsh_set(lsh_indices.begin(), lsh_indices.end());
    std::set<int> gt_set(gt_indices.begin(), gt_indices.end());

    std::set<int> intersection;
    std::set_intersection(lsh_set.begin(), lsh_set.end(), gt_set.begin(), gt_set.end(),
                          std::inserter(intersection, intersection.begin()));

    return static_cast<double>(intersection.size()) / gt_set.size();
}

RecallResults evaluate_recall(const float* Q_all, const float* X, int n_samples, int n_features,
                              int n_eval_queries, const std::vector<size_t>& candidate_counts,
                              const std::vector<size_t>& candidate_offsets,
                              const std::vector<int>& all_candidates, bool verbose) {
    RecallResults results;
    results.n_eval_queries = n_eval_queries;
    results.queries_with_candidates = 0;
    results.avg_recall = 0.0;

    double total_recall = 0.0;

    if (verbose) {
        std::cout << "\nRecall evaluation (" << n_eval_queries << " queries):" << std::endl;
    }

    for (int q = 0; q < n_eval_queries; ++q) {
        size_t count = candidate_counts[q];
        size_t offset = candidate_offsets[q];

        results.per_query_candidates.push_back(count);

        if (count > 0) {
            std::vector<int> lsh_indices(all_candidates.begin() + offset,
                                         all_candidates.begin() + offset + count);

            int gt_size = static_cast<int>(lsh_indices.size());
            std::vector<int> gt_indices =
                get_gt_top_k_indices(&Q_all[q * n_features], X, n_samples, n_features, gt_size);
            double recall = calculate_recall(lsh_indices, gt_indices);

            results.per_query_recall.push_back(recall);
            total_recall += recall;
            results.queries_with_candidates++;

            if (verbose) {
                std::cout << "  Query " << q << ": recall=" << std::fixed << std::setprecision(4)
                          << recall << " (candidates=" << count << ")" << std::endl;
            }
        } else {
            results.per_query_recall.push_back(0.0);
            if (verbose) {
                std::cout << "  Query " << q << ": no candidates" << std::endl;
            }
        }
    }

    results.avg_recall =
        (results.queries_with_candidates > 0) ? total_recall / results.queries_with_candidates : 0.0;

    if (verbose) {
        std::cout << "\nAverage recall: " << std::fixed << std::setprecision(4) << results.avg_recall
                  << " (" << results.queries_with_candidates << "/" << n_eval_queries
                  << " queries with candidates)" << std::endl;
    }

    return results;
}

} // namespace bench

