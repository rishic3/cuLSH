#include <random_projection_lsh.h>

#include <Eigen/Dense>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace Eigen;
using namespace std;
namespace fs = filesystem;

/*
 * Read SIFT .fvecs file into MatrixXd
 */
MatrixXd read_fvecs(const string& filepath) {
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
    int n_vectors = file_size / vector_size;

    MatrixXd data(n_vectors, d);

    // read all vectors into MatrixXd
    for (int i = 0; i < n_vectors; ++i) {
        int dim;
        file.read(reinterpret_cast<char*>(&dim), sizeof(int));
        if (dim != d) {
            throw runtime_error("Inconsistent dimensions in file");
        }

        vector<float> vec(d);
        file.read(reinterpret_cast<char*>(vec.data()), d * sizeof(float));

        for (int j = 0; j < d; ++j) {
            data(i, j) = static_cast<double>(vec[j]);
        }
    }

    return data;
}

vector<int> get_gt_top_k_indices(const VectorXd& q, const MatrixXd& X, int k) {
    VectorXd q_norm = q / q.norm();
    VectorXd cos_sims(X.rows());
    for (MatrixXd::Index i = 0; i < X.rows(); ++i) {
        VectorXd x_norm = X.row(i) / X.row(i).norm();
        cos_sims(i) = x_norm.dot(q_norm);
    }

    vector<pair<double, int>> sim_idx;
    for (int i = 0; i < cos_sims.rows(); ++i) {
        sim_idx.push_back({cos_sims(i), i});
    }

    sort(sim_idx.begin(), sim_idx.end(), greater<pair<double, int>>());

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

struct Config {
    fs::path data_dir;
    int n_hash_tables = 16;
    int n_projections = 4;
    unsigned int seed = random_device{}();
    int n_queries = 100;
    fs::path save_dir = "";
};

void print_usage(const char* program_name) {
    cout << "Usage: " << program_name << " [options]\n"
         << "Options:\n"
         << "  -d, --data-dir        Data directory (default: data/sift)\n"
         << "  -h, --n-hash-tables   Number of hash tables (default: 16)\n"
         << "  -p, --n-projections   Number of projections per table (default: 4)\n"
         << "  -s, --seed            Random seed (default: random)\n"
         << "  -q, --num-queries     Number of test queries (default: 100)\n"
         << "  -o, --save-dir        Directory to save model (optional)\n";
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
    Config conf = parse_args(argc, argv);

    if (!fs::exists(conf.data_dir) || !fs::is_directory(conf.data_dir)) {
        throw runtime_error("Directory '" + conf.data_dir.string() + "' not found");
    }

    MatrixXd X = read_fvecs(conf.data_dir / "sift_base.fvecs");
    MatrixXd Q = read_fvecs(conf.data_dir / "sift_query.fvecs");

    cout << "Data shape: " << X.rows() << "x" << X.cols() << endl;
    cout << "Query shape: " << Q.rows() << "x" << Q.cols() << endl;

    auto Q_test = Q.topRows(conf.n_queries);
    cout << "Using " << conf.n_queries << " test queries" << endl;

    // create model
    RandomProjectionLSH lsh(conf.n_hash_tables, conf.n_projections, conf.seed);

    cout << "\nCreated LSH Model:" << endl;
    cout << "  n_hash_tables: " << lsh.get_n_hash_tables() << endl;
    cout << "  n_projections: " << lsh.get_n_projections() << endl;
    cout << "  seed: " << lsh.get_seed() << endl;
    cout << endl;

    // fit
    cout << "Running fit()..." << endl;
    auto start_time = chrono::high_resolution_clock::now();
    auto model = lsh.fit(X);
    auto fit_time = chrono::high_resolution_clock::now() - start_time;
    auto fit_seconds = chrono::duration_cast<chrono::duration<double>>(fit_time).count();
    cout << "fit() completed in " << fit_seconds << "s" << endl << endl;

    // query
    cout << "Running query()..." << endl;
    start_time = chrono::high_resolution_clock::now();
    auto all_neighbors = model.query_indices(Q_test);
    auto query_time = chrono::high_resolution_clock::now() - start_time;
    auto query_seconds = chrono::duration_cast<chrono::duration<double>>(query_time).count();
    cout << "query() completed in " << query_seconds << "s" << endl << endl;

    // calculate recall for first query
    double recall_score = 0.0;
    int intersection_size = 0;
    int gt_size = 0;

    if (!all_neighbors.empty() && !all_neighbors[0].empty()) {
        vector<int> lsh_indices = all_neighbors[0];
        gt_size = lsh_indices.size();
        vector<int> gt_indices = get_gt_top_k_indices(Q_test.row(0), X, gt_size);
        recall_score = calculate_recall(lsh_indices, gt_indices);

        set<int> lsh_set(lsh_indices.begin(), lsh_indices.end());
        set<int> gt_set(gt_indices.begin(), gt_indices.end());
        set<int> intersection;
        set_intersection(lsh_set.begin(), lsh_set.end(), gt_set.begin(), gt_set.end(),
                         inserter(intersection, intersection.begin()));
        intersection_size = intersection.size();

        cout << "First query recall: " << recall_score << " (" << intersection_size << "/"
             << gt_size << ")" << endl;
    } else {
        cout << "No candidates found for first query" << endl;
    }

    // save report
    if (!fs::create_directories("results") && !fs::exists("results")) {
        throw runtime_error("Failed to create results directory");
    }
    auto now = chrono::system_clock::now();
    auto time_t = chrono::system_clock::to_time_t(now);
    stringstream ss;
    ss << put_time(localtime(&time_t), "%Y%m%d_%H%M%S");
    string report_path = "results/report_h" + to_string(conf.n_hash_tables) + "_p" + to_string(conf.n_projections) + "_" + ss.str() + ".json";

    ofstream report(report_path);
    report << "{\n";
    report << "    \"params\": {\n";
    report << "        \"n_hash_tables\": " << conf.n_hash_tables << ",\n";
    report << "        \"n_projections\": " << conf.n_projections << ",\n";
    report << "        \"seed\": " << lsh.get_seed() << ",\n";
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

    cout << "Report saved to " << report_path << endl;

    // optionally save model
    if (!conf.save_dir.empty()) {
        cout << "Saving model to " << conf.save_dir << "..." << endl;
        model.save(conf.save_dir);
        cout << "Model saved successfully" << endl;
    }
}
