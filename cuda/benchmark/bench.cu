#include "bench_utils.hpp"

#include "../culsh/src/core/candidates.cuh"
#include "../culsh/src/core/index.cuh"
#include "../culsh/src/core/utils.cuh"

#include <chrono>
#include <culsh/rplsh/rplsh.hpp>
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace std;
namespace fs = filesystem;

struct Config {
    fs::path data_dir = "/home/rishic/Code/cu-lsh/data/sift";
    fs::path save_dir = "results";
    int n_hash_tables = 16;
    int n_hashes = 4;
    unsigned int seed = random_device{}();
    int n_queries = 100;
};

void print_usage(const char* program_name) {
    cout << "Usage: " << program_name << " [options]\n"
         << "Options:\n"
         << "  -d, --data-dir        Data directory (default: data/sift)\n"
         << "  -t, --n-hash-tables   Number of hash tables (default: 16)\n"
         << "  -h, --n-hashes        Number of hashes per table (default: 4)\n"
         << "  -s, --seed            Random seed (default: random)\n"
         << "  -q, --num-queries     Number of test queries (default: 100)\n"
         << "  -o, --save-dir        Save directory for results (default: results)\n";
}

Config parse_args(int argc, char* argv[]) {
    Config conf;

    static struct option long_options[] = {
        {"data-dir", required_argument, 0, 'd'},    {"n-hash-tables", required_argument, 0, 't'},
        {"n-hashes", required_argument, 0, 'h'},    {"seed", required_argument, 0, 's'},
        {"num-queries", required_argument, 0, 'q'}, {"save-dir", required_argument, 0, 'o'},
    };

    int c;
    while ((c = getopt_long(argc, argv, "d:t:h:s:q:o:", long_options, nullptr)) != -1) {
        switch (c) {
        case 'd':
            conf.data_dir = optarg;
            break;
        case 't':
            conf.n_hash_tables = atoi(optarg);
            break;
        case 'h':
            conf.n_hashes = atoi(optarg);
            break;
        case 's':
            conf.seed = atoi(optarg);
            break;
        case 'q':
            conf.n_queries = atoi(optarg);
            break;
        case 'o':
            conf.save_dir = optarg;
            break;
        default:
            print_usage(argv[0]);
            exit(1);
        }
    }

    return conf;
}

int main(int argc, char* argv[]) {

    // setup CUDA
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    cout << "Using GPU: " << prop.name << endl;
    cout << "Compute Capability: " << prop.major << "." << prop.minor << endl;
    cout << "Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << endl;
    cout << endl;

    Config conf = parse_args(argc, argv);

    if (!fs::exists(conf.data_dir) || !fs::is_directory(conf.data_dir)) {
        throw runtime_error("Directory '" + conf.data_dir.string() + "' not found");
    }

    // read data
    int n_samples, n_features, n_queries_data, n_features_q;
    float* X = bench::read_fvecs(conf.data_dir / "sift_base.fvecs", n_samples, n_features);
    float* Q_all =
        bench::read_fvecs(conf.data_dir / "sift_query.fvecs", n_queries_data, n_features_q);

    if (n_features != n_features_q) {
        throw runtime_error("Dimension mismatch between fit and query data");
    }

    cout << "Data shape: " << n_samples << "x" << n_features << endl;
    cout << "Query shape: " << n_queries_data << "x" << n_features_q << endl;

    // use only first conf.n_queries queries
    int n_test_queries = min(conf.n_queries, n_queries_data);
    cout << "Using " << n_test_queries << " test queries" << endl;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));

    cout << "\nLSH Model:" << endl;
    cout << "  n_hash_tables: " << conf.n_hash_tables << endl;
    cout << "  n_hashes: " << conf.n_hashes << endl;
    cout << "  seed: " << conf.seed << endl;
    cout << endl;

    // Copy data to GPU
    float* X_gpu;
    float* Q_gpu;
    CUDA_CHECK(cudaMalloc(&X_gpu, static_cast<size_t>(n_samples) * n_features * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(&Q_gpu, static_cast<size_t>(n_test_queries) * n_features * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(X_gpu, X, static_cast<size_t>(n_samples) * n_features * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(Q_gpu, Q_all,
                          static_cast<size_t>(n_test_queries) * n_features * sizeof(float),
                          cudaMemcpyHostToDevice));

    culsh::rplsh::RPLSHParams params{conf.n_hash_tables, conf.n_hashes, conf.seed};

    // fit
    cout << "Running CUDA fit()..." << endl;
    auto start_time = chrono::high_resolution_clock::now();
    culsh::rplsh::Index index =
        culsh::rplsh::fit(cublas_handle, stream, X_gpu, n_samples, n_features, params);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto fit_time = chrono::high_resolution_clock::now() - start_time;
    auto fit_seconds = chrono::duration_cast<chrono::duration<double>>(fit_time).count();
    cout << "-> CUDA fit() completed in " << fit_seconds << "s" << endl << endl;

    CUDA_CHECK(cudaFree(X_gpu));

    cout << "Index size: " << index.size_bytes() / (1024 * 1024) << " MB" << endl;

    // query
    cout << "Running CUDA query()..." << endl;
    start_time = chrono::high_resolution_clock::now();
    culsh::rplsh::Candidates candidates =
        culsh::rplsh::query(cublas_handle, stream, Q_gpu, n_test_queries, index);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto query_time = chrono::high_resolution_clock::now() - start_time;
    auto query_seconds = chrono::duration_cast<chrono::duration<double>>(query_time).count();
    cout << "-> CUDA query() completed in " << query_seconds << "s" << endl << endl;

    CUDA_CHECK(cudaFree(Q_gpu));

    // copy candidate results to host for recall evaluation
    const int n_eval_queries = min(20, n_test_queries);

    vector<size_t> h_candidate_counts(n_test_queries);
    vector<size_t> h_candidate_offsets(n_test_queries + 1);
    CUDA_CHECK(cudaMemcpy(h_candidate_counts.data(), candidates.query_candidate_counts,
                          n_test_queries * sizeof(size_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_candidate_offsets.data(), candidates.query_candidate_offsets,
                          (n_test_queries + 1) * sizeof(size_t), cudaMemcpyDeviceToHost));

    size_t total_candidates = h_candidate_offsets[n_test_queries];
    vector<int> h_all_candidates(total_candidates);
    if (total_candidates > 0) {
        CUDA_CHECK(cudaMemcpy(h_all_candidates.data(), candidates.query_candidate_indices,
                              total_candidates * sizeof(int), cudaMemcpyDeviceToHost));
    }

    // evaluate recall
    bench::RecallResults recall_results =
        bench::evaluate_recall(Q_all, X, n_samples, n_features, n_eval_queries, h_candidate_counts,
                               h_candidate_offsets, h_all_candidates, /*verbose=*/true);

    // save report
    fs::path abs_save_dir = fs::absolute(conf.save_dir);
    fs::create_directories(abs_save_dir);
    if (!fs::exists(abs_save_dir)) {
        throw runtime_error("Failed to create save directory: " + abs_save_dir.string());
    }
    auto now = chrono::system_clock::now();
    auto time_t = chrono::system_clock::to_time_t(now);
    stringstream ss;
    ss << put_time(localtime(&time_t), "%Y%m%d_%H%M%S");
    string report_filename = "report_h" + to_string(conf.n_hash_tables) + "_p" +
                             to_string(conf.n_hashes) + "_" + ss.str() + ".json";
    fs::path report_path = abs_save_dir / report_filename;

    ofstream report(report_path);
    if (!report.is_open()) {
        throw runtime_error("Failed to open report file: " + report_path.string());
    }
    report << "{\n";
    report << "    \"params\": {\n";
    report << "        \"n_hash_tables\": " << conf.n_hash_tables << ",\n";
    report << "        \"n_hashes\": " << conf.n_hashes << ",\n";
    report << "        \"seed\": " << conf.seed << ",\n";
    report << "        \"num_queries\": " << conf.n_queries << "\n";
    report << "    },\n";
    report << "    \"runtimes\": {\n";
    report << "        \"fit_time\": " << fit_seconds << ",\n";
    report << "        \"query_time\": " << query_seconds << "\n";
    report << "    },\n";
    report << "    \"recall_evaluation\": {\n";
    report << "        \"n_eval_queries\": " << recall_results.n_eval_queries << ",\n";
    report << "        \"queries_with_candidates\": " << recall_results.queries_with_candidates
           << ",\n";
    report << "        \"avg_recall\": " << recall_results.avg_recall << ",\n";
    report << "        \"per_query_recall\": [";
    for (size_t i = 0; i < recall_results.per_query_recall.size(); ++i) {
        report << recall_results.per_query_recall[i];
        if (i < recall_results.per_query_recall.size() - 1)
            report << ", ";
    }
    report << "]\n";
    report << "    }\n";
    report << "}\n";
    report.close();

    cout << "Report saved to " << report_path.string() << endl;

    // cleanup
    index.free();
    candidates.free();
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    CUDA_CHECK(cudaStreamDestroy(stream));
    delete[] X;
    delete[] Q_all;
}
