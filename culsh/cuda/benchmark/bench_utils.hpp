#pragma once

#include <string>
#include <vector>

namespace bench {

struct RecallResults {
    int n_eval_queries;
    int queries_with_candidates;
    double avg_recall;
    std::vector<double> per_query_recall;
    std::vector<size_t> per_query_candidates;
};


float* read_fvecs(const std::string& filepath, int& n_vectors, int& dimensions);

std::vector<int> get_gt_top_k_indices(const float* q, const float* X, int n_vectors, int dimensions,
                                      int k);

double calculate_recall(const std::vector<int>& lsh_indices, const std::vector<int>& gt_indices);

RecallResults evaluate_recall(const float* Q_all, const float* X, int n_samples, int n_features,
                              int n_eval_queries, const std::vector<size_t>& candidate_counts,
                              const std::vector<size_t>& candidate_offsets,
                              const std::vector<int>& all_candidates, bool verbose = true);

} // namespace bench

