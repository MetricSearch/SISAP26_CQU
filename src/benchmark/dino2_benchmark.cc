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
#include <numeric>
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

// ---------------------------------------------------------------------------
// HDF5 helpers
// ---------------------------------------------------------------------------
namespace mf_dino2_benchmark_detail {

inline std::pair<hsize_t, hsize_t> get_2d_dataset_shape(hid_t dset) {
  hid_t space = H5Dget_space(dset);
  if (space < 0) throw std::runtime_error("Cannot get dataspace");
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
  if (dset < 0) throw std::runtime_error("Cannot open dataset: " + dset_path);

  auto [rows, cols] = get_2d_dataset_shape(dset);
  if (out_rows) *out_rows = rows;
  if (out_cols) *out_cols = cols;

  std::vector<T> buf(static_cast<size_t>(rows) * static_cast<size_t>(cols));

  hid_t mem_type;
  if constexpr (std::is_same_v<T, float>)        mem_type = H5T_NATIVE_FLOAT;
  else if constexpr (std::is_same_v<T, double>)  mem_type = H5T_NATIVE_DOUBLE;
  else if constexpr (std::is_same_v<T, int32_t>) mem_type = H5T_NATIVE_INT32;
  else if constexpr (std::is_same_v<T, int64_t>) mem_type = H5T_NATIVE_INT64;
  else { H5Dclose(dset); throw std::runtime_error("Unsupported type"); }

  herr_t status = H5Dread(dset, mem_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
  H5Dclose(dset);
  if (status < 0) throw std::runtime_error("H5Dread failed for: " + dset_path);
  return buf;
}

inline std::vector<Vec> rows_to_dense_vectors(const std::vector<float>& buf,
                                              hsize_t rows, hsize_t cols) {
  std::vector<Vec> out;
  out.reserve(static_cast<size_t>(rows));
  for (hsize_t i = 0; i < rows; ++i) {
    Vec v(static_cast<int>(cols));
    const float* p = buf.data() + i * cols;
    for (hsize_t j = 0; j < cols; ++j) v[static_cast<int>(j)] = p[j];
    out.push_back(std::move(v));
  }
  return out;
}

}  // namespace mf_dino2_benchmark_detail

// ---------------------------------------------------------------------------
// Load: only /features, no pre-existing queries or knns
// ---------------------------------------------------------------------------
inline std::vector<Vec> load_features(const std::string& path) {
  hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file < 0) throw std::runtime_error("Cannot open HDF5 file: " + path);

  hsize_t rows = 0, cols = 0;
  std::vector<float> buf =
      mf_dino2_benchmark_detail::read_2d_dataset<float>(file, "features", &rows, &cols);
  H5Fclose(file);

  return mf_dino2_benchmark_detail::rows_to_dense_vectors(buf, rows, cols);
}

// ---------------------------------------------------------------------------
// Brute-force exact k-NN for a single query against the full corpus.
// Uses inner product (== cosine for unit vectors).
// Excludes the query's own index from results.
// ---------------------------------------------------------------------------
inline std::vector<int32_t> brute_force_knn(const std::vector<Vec>& corpus,
                                             const Vec& query,
                                             int32_t query_idx,
                                             int k) {
  const size_t n = corpus.size();

  // Compute all inner products
  std::vector<std::pair<float, int32_t>> scores(n);
  for (size_t i = 0; i < n; ++i) {
    scores[i] = {corpus[i].dot(query), static_cast<int32_t>(i)};
  }

  // Partial sort: top-k+1 by descending inner product
  std::partial_sort(scores.begin(), scores.begin() + k + 1, scores.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });

  // Collect top-k, skipping the query itself
  std::vector<int32_t> result;
  result.reserve(static_cast<size_t>(k));
  for (int i = 0; i < k + 1 && static_cast<int>(result.size()) < k; ++i) {
    if (scores[static_cast<size_t>(i)].second != query_idx) {
      result.push_back(scores[static_cast<size_t>(i)].second);
    }
  }
  return result;
}

int main(int argc, char** argv) {
  try {
    std::string path = "/Volumes/Data/mf_dino2.hdf5";
    if (const char* env_path = std::getenv("MF_DINO2_H5")) path = env_path;
    if (argc >= 2) path = argv[1];

    static constexpr int NUM_QUERIES           = 250;
    static constexpr int K                     = 30;
    static constexpr int NUM_HASH_TABLES       = 4;
    static constexpr int NUM_HASH_BITS         = 16;
    static constexpr int NUM_PROBES_PER_TABLE  = 250;
    static constexpr int_fast8_t PATIENCE      = 1;

    // ---- Load all features ----
    auto load_start = std::chrono::high_resolution_clock::now();
    std::vector<Vec> all_data = load_features(path);
    double load_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::high_resolution_clock::now() - load_start).count();

    std::cout << "data load time=" << load_seconds << "\n";
    std::cout << "Loaded: n=" << all_data.size()
              << " dim=" << (all_data.empty() ? 0 : all_data[0].size()) << "\n";
    if (all_data.empty()) throw std::runtime_error("No data loaded");

    // ---- Select 250 evenly-spaced indices as queries ----
    // We pick them with a stride so they're spread across the dataset, then
    // remove them from the corpus so the index never contains a query.
    const size_t n_total = all_data.size();
    const size_t stride  = n_total / static_cast<size_t>(NUM_QUERIES);

    std::vector<int32_t> query_indices;
    query_indices.reserve(static_cast<size_t>(NUM_QUERIES));
    for (int i = 0; i < NUM_QUERIES; ++i) {
      query_indices.push_back(static_cast<int32_t>(
          static_cast<size_t>(i) * stride));
    }

    // Mark query positions so we can skip them when building the corpus
    std::unordered_set<int32_t> query_set(query_indices.begin(), query_indices.end());

    std::vector<Vec> queries;
    queries.reserve(static_cast<size_t>(NUM_QUERIES));
    for (int32_t idx : query_indices) queries.push_back(all_data[static_cast<size_t>(idx)]);

    // Build corpus = all_data minus the 250 query vectors.
    // Also build a mapping: old index → new (corpus) index, for recall scoring.
    std::vector<Vec> corpus;
    corpus.reserve(n_total - static_cast<size_t>(NUM_QUERIES));
    std::vector<int32_t> old_to_new(n_total, -1);
    for (size_t i = 0; i < n_total; ++i) {
      if (query_set.count(static_cast<int32_t>(i)) == 0) {
        old_to_new[i] = static_cast<int32_t>(corpus.size());
        corpus.push_back(all_data[i]);
      }
    }
    all_data.clear();  // free memory
    all_data.shrink_to_fit();

    std::cout << "corpus size=" << corpus.size()
              << " queries=" << queries.size() << "\n";

    // ---- Brute-force ground truth ----
    std::cout << "Computing brute-force ground truth...\n";
    auto bf_start = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<int32_t>> ground_truth(
        static_cast<size_t>(NUM_QUERIES));
    for (int qi = 0; qi < NUM_QUERIES; ++qi) {
      // Search against corpus (query is already excluded from corpus)
      ground_truth[static_cast<size_t>(qi)] =
          brute_force_knn(corpus, queries[static_cast<size_t>(qi)],
                          /*query_idx=*/-1,  // not in corpus, no self to skip
                          K);
    }

    double bf_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::high_resolution_clock::now() - bf_start).count();
    std::cout << "brute-force time=" << bf_seconds << "\n";

    // ---- Build FALCONN index ----
    LSHConstructionParameters params;
    params.dimension                 = corpus[0].size();
    params.lsh_family                = LSHFamily::CrossPolytope;
    params.distance_function         = DistanceFunction::NegativeInnerProduct;
    params.storage_hash_table        = StorageHashTable::BitPackedFlatHashTable;
    params.l                         = NUM_HASH_TABLES;
    params.num_rotations             = 2;
    params.num_setup_threads         = 8;
    params.feature_hashing_dimension = -1;  // dense
    compute_number_of_hash_functions<Vec>(NUM_HASH_BITS, &params);

    auto table_start = std::chrono::high_resolution_clock::now();
    auto table = construct_table<Vec>(corpus, params);
    double table_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::high_resolution_clock::now() - table_start).count();
    std::cout << "index build time=" << table_seconds << "\n";

    auto query_obj = table->construct_query_object(NUM_PROBES_PER_TABLE);

    // ---- Warm-up pass (not timed, not scored) ----
    {
      std::vector<int32_t> warmup_result;
      for (int qi = 0; qi < NUM_QUERIES; ++qi) {
        warmup_result.clear();
        query_obj->find_k_nearest_neighbours_cqu_lsh(
            queries[static_cast<size_t>(qi)], K, PATIENCE, &warmup_result);
      }
    }

    // ---- Timed query pass + recall accumulation ----
    int64_t total_overlap = 0;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int qi = 0; qi < NUM_QUERIES; ++qi) {
      std::vector<int32_t> result;
      query_obj->find_k_nearest_neighbours_cqu_lsh(
          queries[static_cast<size_t>(qi)], K, PATIENCE, &result);

      // Count how many of the returned IDs are in the ground-truth set
      const std::unordered_set<int32_t> gt_set(
          ground_truth[static_cast<size_t>(qi)].begin(),
          ground_truth[static_cast<size_t>(qi)].end());
      for (int32_t id : result) {
        if (gt_set.count(id)) ++total_overlap;
      }

      if (DEBUG) {
        std::cout << "q=" << qi << " result_size=" << result.size();
        if (!result.empty()) std::cout << " first=" << result[0];
        std::cout << "\n";
      }
    }

    double query_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::high_resolution_clock::now() - t0).count();

    // ---- Report ----
    const double recall =
        static_cast<double>(total_overlap) /
        static_cast<double>(NUM_QUERIES * K);

    std::cout << "total query time=" << query_seconds << "\n";
    std::cout << "mean query time="
              << (query_seconds / static_cast<double>(NUM_QUERIES)) * 1e3
              << " ms\n";
    std::cout << "recall@" << K << "=" << recall
              << " (" << total_overlap << "/" << (NUM_QUERIES * K) << ")\n";

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
