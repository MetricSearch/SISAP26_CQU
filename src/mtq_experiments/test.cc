#include "falconn/lsh_nn_table.h"

#include <Eigen/Dense>

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include<set>

using std::cerr;
using std::cout;
using std::endl;
using std::exception;
using std::fixed;
using std::mt19937_64;
using std::normal_distribution;
using std::scientific;
using std::sqrt;
using std::thread;
using std::uniform_int_distribution;
using std::unique_ptr;
using std::vector;

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;

using falconn::construct_table;
using falconn::DenseVector;
using falconn::DistanceFunction;
using falconn::LSHConstructionParameters;
using falconn::LSHFamily;
using falconn::LSHNearestNeighborQueryPool;
using falconn::LSHNearestNeighborTable;
using falconn::QueryStatistics;
using falconn::StorageHashTable;

typedef falconn::DenseVector<float> Vec;

class Timer {
 public:
  Timer() { start_time = high_resolution_clock::now(); }

  double elapsed_seconds() {
    auto end_time = high_resolution_clock::now();
    auto elapsed = duration_cast<duration<double>>(end_time - start_time);
    return elapsed.count();
  }

 private:
  high_resolution_clock::time_point start_time;
};

std::vector<Vec> load_text_files() {
    std::vector<Vec> data;
    const int dimension = 384;
    
    for (int file_num = 1; file_num <= 10; ++file_num) {
        std::string filename = "/Volumes/Data/mf_dino2_text/" + std::to_string(file_num) + ".txt";
        std::ifstream infile(filename);
        
        if (!infile.is_open()) {
            std::cerr << "Error opening file: " << filename << std::endl;
            continue;
        }
        
        std::string line;
        while (std::getline(infile, line)) {
            std::istringstream iss(line);
            Vec point(dimension);
            
            for (int i = 0; i < dimension; ++i) {
                if (!(iss >> point[i])) {
                    std::cerr << "Error reading dimension " << i << " from " << filename << std::endl;
                    break;
                }
            }
            
            data.push_back(point);
        }
        
        infile.close();
    }
    
    return data;
}

std::vector<Vec> l2_normalize(const std::vector<Vec>& data) {
    std::vector<Vec> normalized_data;
    normalized_data.reserve(data.size());
    
    for (const auto& vec : data) {
        float norm = vec.norm();
        if (norm > 0) {
            normalized_data.push_back(vec / norm);
        } else {
            normalized_data.push_back(vec);
        }
    }
    
    return normalized_data;
}

int main() {

    // Define parameters of search
    StorageHashTable storage_hash_table = StorageHashTable::FlatHashTable;
    DistanceFunction distance_function = DistanceFunction::NegativeInnerProduct;

    // Cross polytope hashing
    LSHConstructionParameters params_cp;
    params_cp.dimension = 384;
    params_cp.lsh_family = LSHFamily::CrossPolytope;
    params_cp.distance_function = distance_function;
    params_cp.storage_hash_table = storage_hash_table;
    params_cp.k = 3;
    params_cp.l = 6;
    params_cp.last_cp_dimension = 16;
    params_cp.num_rotations = 2;
    params_cp.num_setup_threads = 6;
    params_cp.seed = 328648 ^ 833840234;
    int num_probes_cp = 896;

    int k_nearest_neighours = 20;


    // Makes data
    std::vector<Vec> data = load_text_files();
    data = l2_normalize(data);

    cout << "Data dimension: " << (data.empty() ? 0 : data[0].size()) << endl;
    cout << "Number of data points: " << data.size() << endl;


    // Makes the CP Index
    Timer cp_construction;

    unique_ptr<LSHNearestNeighborTable<Vec>> cptable(
        move(construct_table<Vec>(data, params_cp)));

    double cp_construction_time = cp_construction.elapsed_seconds();

    cout << "Construction time: " << cp_construction_time << " seconds" << endl << endl;

    // Run a single query using the first data point
    unique_ptr<falconn::LSHNearestNeighborQuery<Vec>> query_object = 
        cptable->construct_query_object(num_probes_cp);
    
    int qid;
    cout << "Enter query ID: ";
    std::cin >> qid;
    Vec query = data[qid];

    cout << "Running query on first data point..." << endl;

    std::vector<int32_t> result;
    
    // Finds the nearest neighbour of the query
    Timer query_timer;
    query_object->find_k_nearest_neighbors(query, k_nearest_neighours, &result);

    double query_time = query_timer.elapsed_seconds();
    
    cout << "Nearest neighbor index: " << result[1] << endl;
    cout << "Query time: " << query_time << " seconds" << endl;
    
    // Get query statistics
    QueryStatistics stats = query_object->get_query_statistics();
    cout << "Average number of candidates: " << stats.average_num_candidates << endl;
    cout << "Average number of unique candidates: " << stats.average_num_unique_candidates << endl;

    // Compute ground truth by brute force
        cout << "\nComputing ground truth..." << endl;
        Timer gt_timer;
        
        std::vector<std::pair<float, int32_t>> distances;
        distances.reserve(data.size());
        
        for (int32_t i = 0; i < data.size(); ++i) {
            float dist = -query.dot(data[i]);  // Negative inner product
            distances.push_back({dist, i});
        }
        
        std::partial_sort(distances.begin(), 
                         distances.begin() + k_nearest_neighours, 
                         distances.end());
        
        std::vector<int32_t> ground_truth;
        for (int i = 0; i < k_nearest_neighours; ++i) {
            ground_truth.push_back(distances[i].second);
        }
        
        double gt_time = gt_timer.elapsed_seconds();
        cout << "Ground truth computation time: " << gt_time << " seconds" << endl;
        
        // Compute recall for regular query
        std::set<int32_t> gt_set(ground_truth.begin(), ground_truth.end());
        int matches = 0;
        for (int32_t idx : result) {
            if (gt_set.count(idx) > 0) {
                matches++;
            }
        }
        double recall = static_cast<double>(matches) / k_nearest_neighours;
        cout << "\nRegular query recall: " << recall << " (" << matches << "/" << k_nearest_neighours << ")" << endl;
        
        // Compute recall for incremental query
        matches = 0;
        for (int32_t idx : incremental_result) {
            if (gt_set.count(idx) > 0) {
                matches++;
            }
        }
        double incremental_recall = static_cast<double>(matches) / k_nearest_neighours;
        cout << "Incremental query recall: " << incremental_recall << " (" << matches << "/" << k_nearest_neighours << ")" << endl;


        // Print summary of all timings
        cout << "\n=== Timing Summary ===" << endl;
        cout << "Construction time: " << cp_construction_time << " seconds" << endl;
        
        cout << "\nRegular Query:" << endl;
        cout << "  Total time: " << query_time << " seconds" << endl;
        cout << "  LSH time: " << stats.average_lsh_time << " seconds" << endl;
        cout << "  Hash table time: " << stats.average_hash_table_time << " seconds" << endl;
        cout << "  Sketches time: " << stats.average_sketches_time << " seconds" << endl;
        cout << "  Total query time (from stats): " << stats.average_total_query_time << " seconds" << endl;
        
        cout << "\nIncremental Query:" << endl;
        cout << "  Total time: " << incremental_query_time << " seconds" << endl;
        cout << "  LSH time: " << incremental_stats.average_lsh_time << " seconds" << endl;
        cout << "  Hash table time: " << incremental_stats.average_hash_table_time << " seconds" << endl;
        cout << "  Sketches time: " << incremental_stats.average_sketches_time << " seconds" << endl;
        cout << "  Total query time (from stats): " << incremental_stats.average_total_query_time << " seconds" << endl;
        
        cout << "\nGround truth computation time: " << gt_time << " seconds" << endl;
        cout << "\n=== Timing Summary ===" << endl;
        cout << "Construction time: " << cp_construction_time << " seconds" << endl;
        cout << "Regular query time: " << query_time << " seconds" << endl;
        cout << "Incremental query time: " << incremental_query_time << " seconds" << endl;
        cout << "Ground truth computation time: " << gt_time << " seconds" << endl;
}
