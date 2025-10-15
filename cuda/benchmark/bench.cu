#include "../culsh/src/rplsh/candidates.cuh"
#include "../culsh/src/rplsh/index.cuh"
#include "../culsh/src/rplsh/kernels/build_index.cuh"
#include "../culsh/src/rplsh/kernels/hash.cuh"
#include "../culsh/src/rplsh/kernels/projections.cuh"
#include "../culsh/src/rplsh/kernels/query_index.cuh"
#include "../culsh/src/rplsh/utils/utils.cuh"

#include <chrono>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>
using namespace std;
namespace fs = filesystem;

/*
 * Read SIFT .fvecs file
 */
float* read_fvecs(const string& filepath, int& n_vectors, int& dimensions) {
    ifstream file(filepath, ios::binary);
    if (!file.is_open()) {
        throw runtime_error("Failed to open file '" + filepath + "'");
    }

    // read input dimension
    int d;
    file.read(reinterpret_cast<char*>(&d), sizeof(int));

    file.seekg(0, ios::end);
    streamsize file_size = file.tellg();
    file.seekg(0, ios::beg);

    int vector_size = sizeof(int) + d * sizeof(float);
    int n_vecs = file_size / vector_size;

    // allocate memory for data array (row-major: n_vectors x dimensions)
    float* data = new float[n_vecs * d];

    // read all vectors into float array
    for (int i = 0; i < n_vecs; ++i) {
        int dim;
        file.read(reinterpret_cast<char*>(&dim), sizeof(int));
        if (dim != d) {
            throw runtime_error("Inconsistent dimensions in file");
        }

        file.read(reinterpret_cast<char*>(&data[i * d]), d * sizeof(float));
    }

    n_vectors = n_vecs;
    dimensions = d;
    return data;
}

vector<int> get_gt_top_k_indices(const float* q, const float* X, int n_vectors, int dimensions,
                                 int k) {
    // normalize query vector
    float q_norm = 0.0f;
    for (int i = 0; i < dimensions; ++i) {
        q_norm += q[i] * q[i];
    }
    q_norm = sqrt(q_norm);

    vector<float> q_normalized(dimensions);
    for (int i = 0; i < dimensions; ++i) {
        q_normalized[i] = q[i] / q_norm;
    }

    vector<pair<float, int>> sim_idx;
    sim_idx.reserve(n_vectors);

    // compute cosine similarities
    for (int i = 0; i < n_vectors; ++i) {
        // normalize X[i] and compute dot product with normalized query
        float x_norm = 0.0f;
        for (int j = 0; j < dimensions; ++j) {
            float val = X[i * dimensions + j];
            x_norm += val * val;
        }
        x_norm = sqrt(x_norm);

        float cos_sim = 0.0f;
        for (int j = 0; j < dimensions; ++j) {
            cos_sim += (X[i * dimensions + j] / x_norm) * q_normalized[j];
        }

        sim_idx.push_back({cos_sim, i});
    }

    sort(sim_idx.begin(), sim_idx.end(), greater<pair<float, int>>());

    vector<int> top_k;
    for (int i = 0; i < min(k, static_cast<int>(sim_idx.size())); ++i) {
        top_k.push_back(sim_idx[i].second);
    }

    return top_k;
}

double calculate_recall(const vector<int>& lsh_indices, const vector<int>& gt_indices) {
    set<int> lsh_set(lsh_indices.begin(), lsh_indices.end());
    set<int> gt_set(gt_indices.begin(), gt_indices.end());

    set<int> intersection;
    set_intersection(lsh_set.begin(), lsh_set.end(), gt_set.begin(), gt_set.end(),
                     inserter(intersection, intersection.begin()));

    return static_cast<double>(intersection.size()) / gt_set.size();
}

culsh::rplsh::Index cuda_fit(cublasHandle_t cublas_handle, cudaStream_t stream, const float* X_host,
                             int n_samples, int n_features, int n_hash_tables, int n_projections,
                             int seed, float** P_out) {

    // allocate GPU memory for X
    float* X_gpu;
    CUDA_CHECK(cudaMalloc(&X_gpu, static_cast<size_t>(n_samples) * n_features * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(X_gpu, X_host,
                          static_cast<size_t>(n_samples) * n_features * sizeof(float),
                          cudaMemcpyHostToDevice));

    // allocate P on GPU
    float* P;
    int n_total_buckets = n_hash_tables * n_projections;
    CUDA_CHECK(cudaMalloc(&P, static_cast<size_t>(n_total_buckets) * n_features * sizeof(float)));

    // allocate X_hash
    float* X_hash;
    CUDA_CHECK(
        cudaMalloc(&X_hash, static_cast<size_t>(n_samples) * n_total_buckets * sizeof(float)));

    // generate random projections and hash X
    culsh::rplsh::detail::generate_random_projections<float>(stream, n_total_buckets, n_features,
                                                             seed, P);
    culsh::rplsh::detail::hash<float>(cublas_handle, stream, X_gpu, P, n_samples, n_features,
                                      n_hash_tables, n_projections, X_hash);

    // compute binary signatures from X_hash
    int8_t* X_sig;
    CUDA_CHECK(
        cudaMalloc(&X_sig, static_cast<size_t>(n_samples) * n_total_buckets * sizeof(int8_t)));
    culsh::rplsh::detail::compute_signatures<float>(stream, X_hash, n_samples, n_hash_tables,
                                                    n_projections, X_sig);
    CUDA_CHECK(cudaFree(X_hash)); // done with X_hash

    // build and return index
    auto index =
        culsh::rplsh::detail::build_index(stream, X_sig, n_samples, n_hash_tables, n_projections);
    CUDA_CHECK(cudaFree(X_sig));
    CUDA_CHECK(cudaFree(X_gpu));

    *P_out = P; // return projections for later querying
    return index;
}

// CUDA query function (adapted from main.cu)
culsh::rplsh::Candidates cuda_query(cublasHandle_t cublas_handle, cudaStream_t stream,
                                    const float* Q_host, int n_queries, int n_features,
                                    int n_hash_tables, int n_projections, float* P,
                                    culsh::rplsh::Index* index) {

    // allocate GPU memory for Q
    float* Q_gpu;
    CUDA_CHECK(cudaMalloc(&Q_gpu, static_cast<size_t>(n_queries) * n_features * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(Q_gpu, Q_host,
                          static_cast<size_t>(n_queries) * n_features * sizeof(float),
                          cudaMemcpyHostToDevice));

    int n_total_buckets = n_hash_tables * n_projections;

    // allocate Q_hash
    float* Q_hash;
    CUDA_CHECK(
        cudaMalloc(&Q_hash, static_cast<size_t>(n_queries) * n_total_buckets * sizeof(float)));

    // hash queries
    culsh::rplsh::detail::hash<float>(cublas_handle, stream, Q_gpu, P, n_queries, n_features,
                                      n_hash_tables, n_projections, Q_hash);
    CUDA_CHECK(cudaFree(Q_gpu));

    // allocate Q_sig
    int8_t* Q_sig;
    CUDA_CHECK(
        cudaMalloc(&Q_sig, static_cast<size_t>(n_queries) * n_total_buckets * sizeof(int8_t)));

    // convert hash values to signatures
    culsh::rplsh::detail::compute_signatures<float>(stream, Q_hash, n_queries, n_hash_tables,
                                                    n_projections, Q_sig);
    CUDA_CHECK(cudaFree(Q_hash));

    // query index
    culsh::rplsh::Candidates candidates = culsh::rplsh::detail::query_index(
        stream, Q_sig, n_queries, n_hash_tables, n_projections, index);
    CUDA_CHECK(cudaFree(Q_sig));

    return candidates;
}

struct Config {
    fs::path data_dir = "/home/rishic/Code/cu-lsh/data/sift";
    fs::path save_dir = "results";
    int n_hash_tables = 16;
    int n_projections = 4;
    unsigned int seed = random_device{}();
    int n_queries = 100;
};

void print_usage(const char* program_name) {
    cout << "Usage: " << program_name << " [options]\n"
         << "Options:\n"
         << "  -d, --data-dir        Data directory (default: data/sift)\n"
         << "  -h, --n-hash-tables   Number of hash tables (default: 16)\n"
         << "  -p, --n-projections   Number of projections per table (default: 4)\n"
         << "  -s, --seed            Random seed (default: random)\n"
         << "  -q, --num-queries     Number of test queries (default: 100)\n"
         << "  -o, --save-dir        Save directory for results (default: results)\n";
}

Config parse_args(int argc, char* argv[]) {
    Config conf;

    static struct option long_options[] = {
        {"data-dir", required_argument, 0, 'd'},      {"n-hash-tables", required_argument, 0, 'h'},
        {"n-projections", required_argument, 0, 'p'}, {"seed", required_argument, 0, 's'},
        {"num-queries", required_argument, 0, 'q'},   {"save-dir", required_argument, 0, 'o'},
    };

    int c;
    while ((c = getopt_long(argc, argv, "d:h:p:s:q:o:", long_options, nullptr)) != -1) {
        switch (c) {
        case 'd':
            conf.data_dir = optarg;
            break;
        case 'h':
            conf.n_hash_tables = atoi(optarg);
            break;
        case 'p':
            conf.n_projections = atoi(optarg);
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
    float* X = read_fvecs(conf.data_dir / "sift_base.fvecs", n_samples, n_features);
    float* Q_all = read_fvecs(conf.data_dir / "sift_query.fvecs", n_queries_data, n_features_q);

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
    cout << "  n_projections: " << conf.n_projections << endl;
    cout << "  seed: " << conf.seed << endl;
    cout << endl;

    // fit
    cout << "Running CUDA fit()..." << endl;
    auto start_time = chrono::high_resolution_clock::now();
    float* P = nullptr;
    culsh::rplsh::Index index = cuda_fit(cublas_handle, stream, X, n_samples, n_features,
                                         conf.n_hash_tables, conf.n_projections, conf.seed, &P);
    auto fit_time = chrono::high_resolution_clock::now() - start_time;
    auto fit_seconds = chrono::duration_cast<chrono::duration<double>>(fit_time).count();
    cout << "-> CUDA fit() completed in " << fit_seconds << "s" << endl << endl;

    cout << "Index size: " << index.device_size() / (1024 * 1024) << " MB" << endl;

    // query
    cout << "Running CUDA query()..." << endl;
    start_time = chrono::high_resolution_clock::now();
    culsh::rplsh::Candidates candidates =
        cuda_query(cublas_handle, stream, Q_all, n_test_queries, n_features, conf.n_hash_tables,
                   conf.n_projections, P, &index);
    auto query_time = chrono::high_resolution_clock::now() - start_time;
    auto query_seconds = chrono::duration_cast<chrono::duration<double>>(query_time).count();
    cout << "-> CUDA query() completed in " << query_seconds << "s" << endl << endl;

    // calculate recall for first query
    double recall_score = 0.0;
    int intersection_size = 0;
    int gt_size = 0;

    // extract candidates for first query from GPU results
    vector<int> lsh_indices;

    // copy candidate counts for first query to host
    size_t first_query_candidate_count;
    CUDA_CHECK(cudaMemcpy(&first_query_candidate_count, &candidates.query_candidate_counts[0],
                          sizeof(size_t), cudaMemcpyDeviceToHost));

    if (first_query_candidate_count > 0) {
        // get offset for first query candidates
        size_t first_query_offset;
        CUDA_CHECK(cudaMemcpy(&first_query_offset, &candidates.query_candidate_offsets[0],
                              sizeof(size_t), cudaMemcpyDeviceToHost));

        // copy candidate indices for first query to host
        lsh_indices.resize(first_query_candidate_count);
        CUDA_CHECK(cudaMemcpy(lsh_indices.data(),
                              &candidates.query_candidate_indices[first_query_offset],
                              first_query_candidate_count * sizeof(int), cudaMemcpyDeviceToHost));

        if (!lsh_indices.empty()) {
            gt_size = lsh_indices.size();
            vector<int> gt_indices =
                get_gt_top_k_indices(&Q_all[0], X, n_samples, n_features, gt_size);
            recall_score = calculate_recall(lsh_indices, gt_indices);

            set<int> lsh_set(lsh_indices.begin(), lsh_indices.end());
            set<int> gt_set(gt_indices.begin(), gt_indices.end());
            set<int> intersection;
            set_intersection(lsh_set.begin(), lsh_set.end(), gt_set.begin(), gt_set.end(),
                             inserter(intersection, intersection.begin()));
            intersection_size = intersection.size();

            cout << "First query recall: " << recall_score << " (" << intersection_size << "/"
                 << gt_size << ")" << endl;
        }
    } else {
        cout << "No candidates found for first query" << endl;
    }

    // save report
    if (!fs::create_directories(conf.save_dir) && !fs::exists(conf.save_dir)) {
        throw runtime_error("Failed to create save directory: " + conf.save_dir.string());
    }
    auto now = chrono::system_clock::now();
    auto time_t = chrono::system_clock::to_time_t(now);
    stringstream ss;
    ss << put_time(localtime(&time_t), "%Y%m%d_%H%M%S");
    string report_filename = "report_h" + to_string(conf.n_hash_tables) + "_p" +
                         to_string(conf.n_projections) + "_" + ss.str() + ".json";
    fs::path report_path = conf.save_dir / report_filename;

    ofstream report(report_path);
    report << "{\n";
    report << "    \"params\": {\n";
    report << "        \"n_hash_tables\": " << conf.n_hash_tables << ",\n";
    report << "        \"n_projections\": " << conf.n_projections << ",\n";
    report << "        \"seed\": " << conf.seed << ",\n";
    report << "        \"num_queries\": " << conf.n_queries << "\n";
    report << "    },\n";
    report << "    \"runtimes\": {\n";
    report << "        \"fit_time\": " << fit_seconds << ",\n";
    report << "        \"query_time\": " << query_seconds << "\n";
    report << "    },\n";
    report << "    \"first_query_results\": {\n";
    report << "        \"recall_score\": " << recall_score << ",\n";
    report << "        \"intersection_size\": " << intersection_size << ",\n";
    report << "        \"gt_size\": " << gt_size << "\n";
    report << "    }\n";
    report << "}\n";
    report.close();

    cout << "Report saved to " << report_path.string() << endl;

    // cleanup CUDA resources
    CUDA_CHECK(cudaFree(P));
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    CUDA_CHECK(cudaStreamDestroy(stream));

    // cleanup host memory
    delete[] X;
    delete[] Q_all;
}
