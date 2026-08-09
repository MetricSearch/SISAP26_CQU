#include <falconn/lsh_nn_table.h>
#include <hdf5.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#define DEBUG 0

using falconn::DenseVector;
using falconn::DistanceFunction;
using falconn::LSHConstructionParameters;
using falconn::LSHFamily;
using falconn::StorageHashTable;
using falconn::compute_number_of_hash_functions;
using falconn::construct_table;

typedef DenseVector<float> Vec;

namespace task2_benchmark_detail {

inline std::pair<hsize_t, hsize_t> get_2d_dataset_shape(hid_t dset) {
  hid_t space = H5Dget_space(dset);
  if (space < 0) {
    throw std::runtime_error("Cannot get dataspace");
  }
  int ndims = H5Sget_simple_extent_ndims(space);
  if (ndims != 2) {
    H5Sclose(space);
    throw std::runtime_error("Expected a 2D dataset");
  }
  hsize_t dims[2];
  H5Sget_simple_extent_dims(space, dims, nullptr);
  H5Sclose(space);
  return {dims[0], dims[1]};
}

template <typename T>
inline std::vector<T> read_2d_dataset(hid_t file, const std::string& dset_path,
                                      hsize_t* out_rows, hsize_t* out_cols) {
  hid_t dset = H5Dopen2(file, dset_path.c_str(), H5P_DEFAULT);
  if (dset < 0) {
    throw std::runtime_error("Cannot open dataset: " + dset_path);
  }

  auto [rows, cols] = get_2d_dataset_shape(dset);
  if (out_rows) *out_rows = rows;
  if (out_cols) *out_cols = cols;

  std::vector<T> buf(static_cast<size_t>(rows) * static_cast<size_t>(cols));

  hid_t mem_type;
  if constexpr (std::is_same_v<T, float>) {
    mem_type = H5T_NATIVE_FLOAT;
  } else if constexpr (std::is_same_v<T, double>) {
    mem_type = H5T_NATIVE_DOUBLE;
  } else if constexpr (std::is_same_v<T, int32_t>) {
    mem_type = H5T_NATIVE_INT32;
  } else if constexpr (std::is_same_v<T, int64_t>) {
    mem_type = H5T_NATIVE_INT64;
  } else {
    H5Dclose(dset);
    throw std::runtime_error("Unsupported type in read_2d_dataset");
  }

  herr_t status =
      H5Dread(dset, mem_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
  H5Dclose(dset);
  if (status < 0) {
    throw std::runtime_error("H5Dread failed for dataset: " + dset_path);
  }
  return buf;
}

inline std::vector<Vec> rows_to_dense_vectors(const std::vector<float>& buf,
                                              hsize_t rows, hsize_t cols) {
  std::vector<Vec> out;
  out.reserve(static_cast<size_t>(rows));

  for (hsize_t i = 0; i < rows; ++i) {
    Vec v(static_cast<int>(cols));
    const float* row_ptr =
        buf.data() + static_cast<size_t>(i) * static_cast<size_t>(cols);
    for (hsize_t j = 0; j < cols; ++j) {
      v[static_cast<int>(j)] = row_ptr[j];
    }
    out.push_back(std::move(v));
  }
  return out;
}

inline std::vector<float> columnwise_median(const std::vector<Vec>& data) {
  if (data.empty()) {
    return {};
  }
  const size_t n = data.size();
  const int d = data[0].size();

  std::vector<float> med(static_cast<size_t>(d));
  std::vector<float> col(n);

  for (int j = 0; j < d; ++j) {
    for (size_t i = 0; i < n; ++i) {
      col[i] = data[i][j];
    }

    const size_t mid = n / 2;
    std::nth_element(col.begin(), col.begin() + static_cast<long>(mid),
                     col.end());
    const float upper_mid = col[mid];

    if ((n % 2) == 1) {
      med[static_cast<size_t>(j)] = upper_mid;
    } else {
      std::nth_element(col.begin(),
                       col.begin() + static_cast<long>(mid - 1), col.end());
      const float lower_mid = col[mid - 1];
      med[static_cast<size_t>(j)] = 0.5f * (lower_mid + upper_mid);
    }
  }

  return med;
}

inline void subtract_in_place(std::vector<Vec>* points,
                              const std::vector<float>& shift) {
  if (!points) return;
  if (points->empty()) return;
  const int d = (*points)[0].size();
  if (static_cast<int>(shift.size()) != d) {
    throw std::runtime_error("Shift vector size mismatch");
  }

  for (Vec& v : *points) {
    if (v.size() != d) {
      throw std::runtime_error("Inconsistent vector dimensionality");
    }
    for (int j = 0; j < d; ++j) {
      v[j] -= shift[static_cast<size_t>(j)];
    }
  }
}

}  // namespace task2_benchmark_detail

struct DenseBundle {
  std::vector<Vec> data;
  std::vector<Vec> queries;
  std::vector<std::vector<int32_t>> knns;
};

inline DenseBundle load_data(const std::string& path) {
  hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file < 0) {
    throw std::runtime_error("Cannot open HDF5 file: " + path);
  }

  hsize_t train_rows = 0, train_cols = 0;
  std::vector<float> train_buf =
      task2_benchmark_detail::read_2d_dataset<float>(file, "train", &train_rows,
                                                     &train_cols);
  std::vector<Vec> data = task2_benchmark_detail::rows_to_dense_vectors(
      train_buf, train_rows, train_cols);

  hsize_t query_rows = 0, query_cols = 0;
  std::vector<float> query_buf = task2_benchmark_detail::read_2d_dataset<float>(
      file, "test/queries", &query_rows, &query_cols);
  if (query_cols != train_cols) {
    H5Fclose(file);
    throw std::runtime_error(
        "train and test/queries dimensionality mismatch");
  }
  std::vector<Vec> queries = task2_benchmark_detail::rows_to_dense_vectors(
      query_buf, query_rows, query_cols);

  hsize_t knn_rows = 0, knn_cols = 0;
  std::vector<int32_t> knn_buf =
      task2_benchmark_detail::read_2d_dataset<int32_t>(file, "test/knns",
                                                       &knn_rows, &knn_cols);
  H5Fclose(file);

  static constexpr int k = 30;
  if (knn_cols < static_cast<hsize_t>(k)) {
    throw std::runtime_error("test/knns has fewer than 30 columns");
  }
  if (knn_rows != query_rows) {
    throw std::runtime_error("test/knns row count != queries row count");
  }

  std::vector<std::vector<int32_t>> knns(static_cast<size_t>(knn_rows),
                                         std::vector<int32_t>(k));
  for (hsize_t i = 0; i < knn_rows; ++i) {
    for (int j = 0; j < k; ++j) {
      knns[static_cast<size_t>(i)][static_cast<size_t>(j)] =
          knn_buf[static_cast<size_t>(i) * static_cast<size_t>(knn_cols) +
                  static_cast<size_t>(j)];
    }
  }

  return {std::move(data), std::move(queries), std::move(knns)};
}

inline std::pair<std::vector<Vec>, float> build_euclidean_mips_index(
    const std::vector<Vec>& X) {
  if (X.empty()) {
    return {{}, 0.0f};
  }
  const int d = X[0].size();

  float R = 0.0f;
  std::vector<float> norms;
  norms.reserve(X.size());
  for (const Vec& v : X) {
    if (v.size() != d) {
      throw std::runtime_error("Inconsistent vector dimensionality");
    }
    float n = v.norm();
    norms.push_back(n);
    R = std::max(R, n);
  }

  const float R2 = R * R;
  std::vector<Vec> X_transformed;
  X_transformed.reserve(X.size());
  for (size_t i = 0; i < X.size(); ++i) {
    const float n2 = norms[i] * norms[i];
    const float extra =
        static_cast<float>(std::sqrt(std::max(0.0f, R2 - n2)));

    Vec v2(d + 1);
    v2.head(d) = X[i];
    v2[d] = extra;
    X_transformed.push_back(std::move(v2));
  }

  return {std::move(X_transformed), R};
}

inline std::vector<Vec> transform_queries(const std::vector<Vec>& Q) {
  if (Q.empty()) {
    return {};
  }
  const int d = Q[0].size();
  std::vector<Vec> out;
  out.reserve(Q.size());
  for (const Vec& q : Q) {
    if (q.size() != d) {
      throw std::runtime_error("Inconsistent query dimensionality");
    }
    Vec q2(d + 1);
    q2.head(d) = q;
    q2[d] = 0.0f;
    out.push_back(std::move(q2));
  }
  return out;
}

inline DenseBundle load_hyperspherical_data(const std::string& path) {
  DenseBundle bundle = load_data(path);

  // Match the Python exactly:
  //   data -= median(data)
  //   queries -= median(data)   (median computed AFTER the line above)
  // Since median is translation-invariant, the second median is ~0, so queries
  // are (almost) unchanged.
  const std::vector<float> shift1 =
      task2_benchmark_detail::columnwise_median(bundle.data);
  task2_benchmark_detail::subtract_in_place(&bundle.data, shift1);

  const std::vector<float> shift2 =
      task2_benchmark_detail::columnwise_median(bundle.data);
  task2_benchmark_detail::subtract_in_place(&bundle.queries, shift2);

  auto [data_transformed, R] = build_euclidean_mips_index(bundle.data);
  (void)R;
  bundle.data = std::move(data_transformed);
  bundle.queries = transform_queries(bundle.queries);
  return bundle;
}

int main(int argc, char** argv) {
  try {
    std::string path = "/Volumes/Data/llama-sisap-26/llama-dev.h5";
    if (const char* env_path = std::getenv("LLAMA_DEV_H5")) {
      path = env_path;
    }
    if (argc >= 2) {
      path = argv[1];
    }

    auto load_start = std::chrono::high_resolution_clock::now();


    auto [data, queries, knns] = load_hyperspherical_data(path);

    double load_seconds =
    std::chrono::duration_cast<std::chrono::duration<double>>(
        std::chrono::high_resolution_clock::now() - load_start)
        .count();
    std::cout << "data load time=" << load_seconds
              << "\n";

    std::cout << "Loaded: data=" << data.size() << " queries=" << queries.size()
              << " knns=" << knns.size();
    if (!knns.empty()) {
      std::cout << "x" << knns[0].size();
    }
    std::cout << "\n";
    if (!data.empty()) {
      std::cout << "Dim (after transform): " << data[0].size() << "\n";
    }
    if (data.empty() || queries.empty()) {
      throw std::runtime_error("Loaded empty data/queries");
    }

    // ------------------------------------------------------------
    // Minimal CQU-LSH call (tweak constants as needed)
    // ------------------------------------------------------------
    static constexpr int NUM_HASH_TABLES = 15;
    static constexpr int NUM_HASH_BITS = 16;
    static constexpr int NUM_PROBES_PER_TABLE = 115;
    static constexpr int K = 30;
    static constexpr int_fast8_t PATIENCE = 4;

    LSHConstructionParameters params;
    params.dimension = data[0].size();
    params.lsh_family = LSHFamily::CrossPolytope;
    params.distance_function = DistanceFunction::NegativeInnerProduct;
    params.storage_hash_table = StorageHashTable::BitPackedFlatHashTable;
    params.l = NUM_HASH_TABLES;
    params.num_rotations = 2;
    params.num_setup_threads = 8;
    params.feature_hashing_dimension = -1;  // dense
    compute_number_of_hash_functions<Vec>(NUM_HASH_BITS, &params);

    auto table_start = std::chrono::high_resolution_clock::now();

    auto table = construct_table<Vec>(data, params);

    double table_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::high_resolution_clock::now() - table_start)
            .count();
    std::cout << "index build time=" << table_seconds
              << "\n";

    auto query_obj = table->construct_query_object(NUM_PROBES_PER_TABLE);

    std::vector<int32_t> result;
    const int num_test_queries =
      std::min<int>(1000, static_cast<int>(queries.size()));


    int64_t total_overlap = 0;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int _i = 0; _i < 10; _i++) {

    
    for (int qi = 0; qi < num_test_queries; ++qi) {

      if (DEBUG) {
        std::cout << "query[" << qi << "][0]=" << queries[qi][0] << "\n";
      }

      result.clear();
      query_obj->find_k_nearest_neighbours_cqu_lsh(queries[qi], K, PATIENCE,
                                                   &result);

      if (DEBUG) {
        std::cout << "q=" << qi << " -> result size=" << result.size();
        if (!result.empty()) {
            std::cout << " first=" << result[0];
            if (result.size() > 1) {
            std::cout << " second=" << result[1];
            }
        }
      }
    }
    }

    double seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::high_resolution_clock::now() - t0)
            .count();
    std::cout << "total query time=" << seconds
              << "\n";


  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
