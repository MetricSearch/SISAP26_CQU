#include "falconn/lsh_nn_table.h"

#include <Eigen/Dense>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>
#include <fstream>

using std::cerr;
using std::cout;
using std::endl;
using std::vector;

using falconn::DenseVector;
using falconn::DistanceFunction;
using falconn::LSHConstructionParameters;
using falconn::LSHFamily;
using falconn::LSHNearestNeighborQueryPool;
using falconn::LSHNearestNeighborTable;
using falconn::StorageHashTable;

typedef falconn::DenseVector<float> Vec;
typedef Eigen::MatrixXf Matrix;

// Load matrix from binary file
// Format: first uint64 is rows, second uint64 is cols, then row*cols floats
template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> load_matrix_from_binary(
    const std::string& filename) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file: " + filename);
  }
  
  uint64_t rows, cols;
  file.read(reinterpret_cast<char*>(&rows), sizeof(uint64_t));
  file.read(reinterpret_cast<char*>(&cols), sizeof(uint64_t));
  
  cout << "Loading matrix from " << filename << " with shape [" 
       << rows << ", " << cols << "]" << endl;
  
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> matrix(rows, cols);
  
  std::vector<Scalar> buffer(rows * cols);
  file.read(reinterpret_cast<char*>(buffer.data()), rows * cols * sizeof(Scalar));
  
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      matrix(i, j) = buffer[i * cols + j];
    }
  }
  
  return matrix;
}

int main() {
  try {
    cout << "Loading data from binary files..." << endl;
    
    std::string data_path = "/tmp/final_data.bin";
    std::string queries_path = "/tmp/final_queries.bin";
    
    // Load matrices
    auto final_data = load_matrix_from_binary<float>(data_path);
    auto final_queries = load_matrix_from_binary<float>(queries_path);
    
    cout << "Data shape: [" << final_data.rows() << ", " << final_data.cols() << "]" << endl;
    cout << "Queries shape: [" << final_queries.rows() << ", " << final_queries.cols() << "]" << endl;
    
    // Convert Eigen matrices to FALCONN format
    uint32_t n = final_data.rows();
    uint32_t d = final_data.cols();
    
    cout << "\nSetting up FALCONN parameters..." << endl;
    cout << "n = " << n << ", d = " << d << endl;
    
    // Convert to vector of DenseVector
    vector<Vec> data(n);
    for (uint32_t i = 0; i < n; ++i) {
      data[i].resize(d);
      for (uint32_t j = 0; j < d; ++j) {
        data[i][j] = final_data(i, j);
      }
    }
    
    // Convert query to DenseVector
    Vec query(d);
    for (uint32_t j = 0; j < d; ++j) {
      query[j] = final_queries(0, j);
    }
    
    // Set up FALCONN parameters (same as notebook)
    LSHConstructionParameters params;
    params.dimension = d;
    params.lsh_family = LSHFamily::CrossPolytope;
    params.distance_function = DistanceFunction::EuclideanSquared;
    params.storage_hash_table = StorageHashTable::BitPackedFlatHashTable;
    params.k = 2;
    params.l = 32;
    params.num_rotations = 1;
    params.num_setup_threads = 0;  // Use all cores
    params.last_cp_dimension = 64;  // 128
    
    cout << "\nFALCONN Parameters:" << endl;
    cout << "  k = " << params.k << endl;
    cout << "  l = " << params.l << endl;
    cout << "  last_cp_dimension = " << params.last_cp_dimension << endl;
    cout << "  num_rotations = " << params.num_rotations << endl;
    
    cout << "\nBuilding LSH index..." << endl;
    auto index = falconn::construct_table<Vec>(data, params);
    cout << "Index built successfully!" << endl;
    
    cout << "\nConstructing query object..." << endl;
    uint32_t probes_per_table = 16;
    auto query_object = index->construct_query_object(probes_per_table);
    cout << "Query object constructed." << endl;
    
    // Run query with find_k_nearest_neighbours_mtq
    cout << "\nRunning find_k_nearest_neighbours_mtq..." << endl;
    int_fast64_t k = 30;
    int_fast64_t warmup = 2;
    
    cout << "  Query k = " << k << endl;
    cout << "  Warmup = " << warmup << endl;
    cout << "  Probes per table = " << probes_per_table << endl;
    
    // Test with multiple queries
    int num_test_queries = 5;
    cout << "\nTesting with " << num_test_queries << " queries from loaded data:" << endl;
    
    for (int qid = 0; qid < num_test_queries && qid < final_queries.rows(); ++qid) {
      // Convert query to DenseVector
      Vec test_query(d);
      for (uint32_t j = 0; j < d; ++j) {
        test_query[j] = final_queries(qid, j);
      }
      
      std::vector<int> neighbors;
      query_object->find_k_nearest_neighbours_mtq(test_query, k, warmup, &neighbors);
      
      cout << "Query " << qid << ": found " << neighbors.size() << " neighbors";
      if (neighbors.size() > 0) {
        cout << " (first 5: " << neighbors[0];
        for (size_t i = 1; i < std::min(size_t(5), neighbors.size()); ++i) {
          cout << ", " << neighbors[i];
        }
        cout << ")";
      }
      cout << endl;
    }
    
    cout << "\nAll queries completed successfully!" << endl;
    
  } catch (const std::exception& e) {
    cerr << "Exception caught: " << e.what() << endl;
    return 1;
  }
  
  return 0;
}
