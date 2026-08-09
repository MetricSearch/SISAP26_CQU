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
#include <set>

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
    
    for (int file_num = 1; file_num <= 1000; ++file_num) {
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
    params_cp.k = 2;
    params_cp.l = 12;
    params_cp.last_cp_dimension = 384;
    params_cp.num_rotations = 2;
    params_cp.num_setup_threads = 8;
    params_cp.seed = 328648 ^ 833840234;
    int num_probes_cp = 384;

    int k_nearest_neighours = 20;
    int num_queries = 100;
    int warmup = 100;

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

    std::this_thread::sleep_for(std::chrono::seconds(10));

    unique_ptr<falconn::LSHNearestNeighborQuery<Vec>> query_standard = 
        cptable->construct_query_object(num_probes_cp);
    
    std::vector<int32_t> result_standard;
    Timer standard_timer;
    
    for (int i = 0; i < num_queries; ++i) {
        int qid = i % data.size();
        Vec query = data[qid];
        query_standard->find_k_nearest_neighbors(query, k_nearest_neighours, &result_standard);
    }

    double standard_time = standard_timer.elapsed_seconds();
    
    cout << "Time for " << num_queries << " queries: " << (standard_time * 1000) << " ms" << endl;
    cout << "Average time per query: " << (standard_time / num_queries * 1000) << " ms" << endl;

    
    return 0;
}
