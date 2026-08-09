#ifndef __NN_QUERY_H__
#define __NN_QUERY_H__

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../falconn_global.h"
#include "heap.h"


namespace falconn
{
  namespace core
  {

    class NearestNeighborQueryError : public FalconnError
    {
    public:
      NearestNeighborQueryError(const char *msg) : FalconnError(msg) {}
    };

    template <typename LSHTableQuery, typename LSHTablePointType,
              typename LSHTableKeyType, typename ComparisonPointType,
              typename DistanceType, typename DistanceFunction,
              typename DataStorage>
    class NearestNeighborQuery
    {
    public:
      NearestNeighborQuery(LSHTableQuery *table_query,
                           const DataStorage &data_storage)
          : table_query_(table_query), data_storage_(data_storage) {}
      LSHTableKeyType find_nearest_neighbor(const LSHTablePointType &q,
                                            const ComparisonPointType &q_comp,
                                            int_fast64_t num_probes,
                                            int_fast64_t max_num_candidates)
      {
        auto start_time = std::chrono::high_resolution_clock::now();

        table_query_->get_unique_candidates(q, num_probes, max_num_candidates,
                                            &candidates_);
        auto distance_start_time = std::chrono::high_resolution_clock::now();

        LSHTableKeyType best_key = -1;

        if (candidates_.size() > 0)
        {
          typename DataStorage::SubsequenceIterator iter =
              data_storage_.get_subsequence(candidates_);

          best_key = candidates_[0];
          DistanceType best_distance = dst_(q_comp, iter.get_point());
          ++iter;

          while (iter.is_valid())
          {
            DistanceType cur_distance = dst_(q_comp, iter.get_point());
            if (cur_distance < best_distance)
            {
              best_distance = cur_distance;
              best_key = iter.get_key();
            }
            ++iter;
          }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_distance =
            std::chrono::duration_cast<std::chrono::duration<double>>(
                end_time - distance_start_time);
        auto elapsed_total =
            std::chrono::duration_cast<std::chrono::duration<double>>(end_time -
                                                                      start_time);
        stats_.average_distance_time += elapsed_distance.count();
        stats_.average_total_query_time += elapsed_total.count();

        return best_key;
      }

      void find_k_nearest_neighbors(const LSHTablePointType &q,
                                    const ComparisonPointType &q_comp,
                                    int_fast64_t k, int_fast64_t num_probes,
                                    int_fast64_t max_num_candidates,
                                    std::vector<LSHTableKeyType> *result)
      {

        if (result == nullptr)
        {
          throw NearestNeighborQueryError("Results vector pointer is nullptr.");
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        std::vector<LSHTableKeyType> &res = *result;
        res.clear();

        table_query_->get_unique_candidates(q, num_probes, max_num_candidates,
                                            &candidates_);

        heap_.reset();
        heap_.resize(k);

        auto distance_start_time = std::chrono::high_resolution_clock::now();

        typename DataStorage::SubsequenceIterator iter =
            data_storage_.get_subsequence(candidates_);

        int_fast64_t initially_inserted = 0;
        for (; initially_inserted < k; ++initially_inserted)
        {
          if (iter.is_valid())
          {
            heap_.insert_unsorted(-dst_(q_comp, iter.get_point()), iter.get_key());
            ++iter;
          }
          else
          {
            break;
          }
        }

        if (initially_inserted >= k)
        {
          heap_.heapify();
          while (iter.is_valid())
          {
            DistanceType cur_distance = dst_(q_comp, iter.get_point());
            if (cur_distance < -heap_.min_key())
            {
              heap_.replace_top(-cur_distance, iter.get_key());
            }
            ++iter;
          }
        }

        res.resize(initially_inserted);
        std::sort(heap_.get_data().begin(),
                  heap_.get_data().begin() + initially_inserted);
        for (int_fast64_t ii = 0; ii < initially_inserted; ++ii)
        {
          res[ii] = heap_.get_data()[initially_inserted - ii - 1].data;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_distance =
            std::chrono::duration_cast<std::chrono::duration<double>>(
                end_time - distance_start_time);
        auto elapsed_total =
            std::chrono::duration_cast<std::chrono::duration<double>>(end_time -
                                                                      start_time);
        stats_.average_distance_time += elapsed_distance.count();
        stats_.average_total_query_time += elapsed_total.count();
      }

      template <typename PointType>
      static auto clear_vector(PointType *point)
          -> decltype((*point) *= 0, void())
      {
        (*point) *= 0;
      }

      template <typename IndexType, typename CoordinateType>
      static void clear_vector(
          std::vector<std::pair<IndexType, CoordinateType>> *point)
      {
        point->clear();
      }

      template <typename AccumulatorType, typename PointType>
      static auto add_to_vector(AccumulatorType *accumulator,
                                const PointType &point)
          -> decltype((*accumulator) += point, void())
      {
        (*accumulator) += point;
      }

      template <typename AccumulatorType, typename PointType>
      static auto subtract_from_vector(AccumulatorType *accumulator,
                                       const PointType &point)
          -> decltype((*accumulator) -= point, void())
      {
        (*accumulator) -= point;
      }

      template <typename IndexType, typename CoordinateType>
      static void combine_sparse_vector(
          std::vector<std::pair<IndexType, CoordinateType>> *accumulator,
          const std::vector<std::pair<IndexType, CoordinateType>> &point,
          CoordinateType scale)
      {
        std::vector<std::pair<IndexType, CoordinateType>> merged;
        merged.reserve(accumulator->size() + point.size());
        size_t acc_idx = 0;
        size_t point_idx = 0;

        while (acc_idx < accumulator->size() && point_idx < point.size())
        {
          const auto &acc_value = (*accumulator)[acc_idx];
          const auto &point_value = point[point_idx];
          if (acc_value.first < point_value.first)
          {
            merged.push_back(acc_value);
            ++acc_idx;
          }
          else if (point_value.first < acc_value.first)
          {
            merged.push_back(
                {point_value.first, scale * point_value.second});
            ++point_idx;
          }
          else
          {
            const CoordinateType value =
                acc_value.second + scale * point_value.second;
            if (value != CoordinateType{})
            {
              merged.push_back({acc_value.first, value});
            }
            ++acc_idx;
            ++point_idx;
          }
        }

        merged.insert(merged.end(), accumulator->begin() + acc_idx,
                      accumulator->end());
        for (; point_idx < point.size(); ++point_idx)
        {
          merged.push_back(
              {point[point_idx].first, scale * point[point_idx].second});
        }
        accumulator->swap(merged);
      }

      template <typename IndexType, typename CoordinateType>
      static void add_to_vector(
          std::vector<std::pair<IndexType, CoordinateType>> *accumulator,
          const std::vector<std::pair<IndexType, CoordinateType>> &point)
      {
        combine_sparse_vector(accumulator, point,
                              static_cast<CoordinateType>(1));
      }

      template <typename IndexType, typename CoordinateType>
      static void subtract_from_vector(
          std::vector<std::pair<IndexType, CoordinateType>> *accumulator,
          const std::vector<std::pair<IndexType, CoordinateType>> &point)
      {
        combine_sparse_vector(accumulator, point,
                              static_cast<CoordinateType>(-1));
      }

      void recompute_centroid(ComparisonPointType *q_prime, int_fast64_t k)
      {
        std::vector<LSHTableKeyType> heap_keys;
        heap_keys.reserve(k);
        for (int_fast64_t ii = 0; ii < k; ++ii)
        {
          heap_keys.push_back(heap_.get_data()[ii].data);
        }

        typename DataStorage::SubsequenceIterator iter =
            data_storage_.get_subsequence(heap_keys);
        clear_vector(q_prime);
        while (iter.is_valid())
        {
          add_to_vector(q_prime, iter.get_point());
          ++iter;
        }
      }

      void update_centroid(ComparisonPointType *q_prime,
                           bool centroid_is_initialized, int_fast64_t k)
      {
        const int_fast64_t num_changes =
            static_cast<int_fast64_t>(old_keys_.size() + new_keys_.size());
        if (!centroid_is_initialized || num_changes > k)
        {
          recompute_centroid(q_prime, k);
          return;
        }

        typename DataStorage::SubsequenceIterator added_iter =
            data_storage_.get_subsequence(new_keys_);
        while (added_iter.is_valid())
        {
          add_to_vector(q_prime, added_iter.get_point());
          ++added_iter;
        }

        typename DataStorage::SubsequenceIterator removed_iter =
            data_storage_.get_subsequence(old_keys_);
        while (removed_iter.is_valid())
        {
          subtract_from_vector(q_prime, removed_iter.get_point());
          ++removed_iter;
        }
      }

      void record_heap_change(
          const typename DataStorage::SubsequenceIterator &iter)
      {
        auto added =
            std::find(new_keys_.begin(), new_keys_.end(), heap_.min_data());
        if (added != new_keys_.end())
        {
          *added = iter.get_key();
          return;
        }

        old_keys_.push_back(heap_.min_data());
        new_keys_.push_back(iter.get_key());
      }

      void find_k_nearest_neighbours_cqu_lsh(
          const LSHTablePointType &q, const ComparisonPointType &q_comp,
          int_fast64_t k,
          int_fast64_t num_probes_per_table,
          int_fast64_t max_num_candidates,
          std::vector<LSHTableKeyType> *result,
          int_fast64_t patience)
      {
        if (result == nullptr)
        {
          throw NearestNeighborQueryError("Results vector pointer is nullptr.");
        }
        table_query_->start_new_query();

        auto start_time = std::chrono::high_resolution_clock::now();

        std::vector<LSHTableKeyType> &res = *result;
        res.clear();

        heap_.reset();
        heap_.resize(k);
        ComparisonPointType q_prime = q_comp;

        auto distance_start_time = std::chrono::high_resolution_clock::now();

        const int_fast64_t num_tables = table_query_->get_num_tables();
        int_fast64_t initially_inserted = 0;
        int_fast64_t last_successful_table = -1;
        int_fast64_t table_idx = 0;

        bool centroid_is_initialized = false;
        old_keys_.clear();
        new_keys_.clear();

        while (table_idx <= num_tables ||
               table_idx - last_successful_table <= patience)
        {
          candidates_.clear();
          table_query_->get_candidates_at_table(q_prime,
                                                table_idx % num_tables,
                                                num_probes_per_table,
                                                max_num_candidates,
                                                &candidates_);

          if (candidates_.empty())
          {
            ++table_idx;
            continue;
          }

          typename DataStorage::SubsequenceIterator iter =
              data_storage_.get_subsequence(candidates_);

          while (iter.is_valid())
          {
            DistanceType cur_distance = dst_(q_comp, iter.get_point());
            if (initially_inserted < k)
            {
              heap_.insert_unsorted(-cur_distance, iter.get_key());
              ++initially_inserted;
              if (initially_inserted == k)
              {
                heap_.heapify();
              }
            }
            else if (cur_distance < -heap_.min_key())
            {
              last_successful_table = table_idx;
              record_heap_change(iter);
              heap_.replace_top(-cur_distance, iter.get_key());
            }
            ++iter;
          }

          if (initially_inserted >= k && table_idx >= num_tables - 1)
          {
            update_centroid(&q_prime, centroid_is_initialized, k);
            centroid_is_initialized = true;
          }

          old_keys_.clear();
          new_keys_.clear();

          ++table_idx;
        }

        res.resize(initially_inserted);

        // Sorts the heap from start to the kth thing
        std::sort(heap_.get_data().begin(),
                  heap_.get_data().begin() + initially_inserted);
        for (int_fast64_t ii = 0; ii < initially_inserted; ++ii)
        {
          res[ii] = heap_.get_data()[initially_inserted - ii - 1].data;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_distance =
            std::chrono::duration_cast<std::chrono::duration<double>>(
                end_time - distance_start_time);
        auto elapsed_total =
            std::chrono::duration_cast<std::chrono::duration<double>>(
                end_time - start_time);
        stats_.average_distance_time += elapsed_distance.count();
        stats_.average_total_query_time += elapsed_total.count();
      }

      void find_k_nearest_neighbours_mtq(
          const LSHTablePointType &q, const ComparisonPointType &q_comp,
          int_fast64_t k,
          int_fast64_t num_probes_per_table,
          int_fast64_t max_num_candidates,
          int_fast64_t warmup,
          std::vector<LSHTableKeyType> *result)
      {

        if (result == nullptr)
        {
          throw NearestNeighborQueryError("Results vector pointer is nullptr.");
        }

        table_query_->start_new_query();

        auto start_time = std::chrono::high_resolution_clock::now();


        std::vector<LSHTableKeyType> &res = *result;
        res.clear();

        heap_.reset();
        heap_.resize(k);
        ComparisonPointType q_prime = q_comp;

        auto distance_start_time = std::chrono::high_resolution_clock::now();

        const int_fast64_t num_tables = table_query_->get_num_tables();

        int_fast64_t initially_inserted = 0;
        int_fast64_t num_seen = 0;
        bool centroid_is_initialized = false;

        old_keys_.clear();
        new_keys_.clear();

        for (int table_idx = 0; table_idx < num_tables; ++table_idx)
        {
          candidates_.clear();
          table_query_->get_candidates_at_table(
              q_prime, table_idx, num_probes_per_table, max_num_candidates,
              &candidates_);

          if (candidates_.empty())
          {
            continue;
          }

          typename DataStorage::SubsequenceIterator iter =
              data_storage_.get_subsequence(candidates_);

          while (iter.is_valid())
          {
            DistanceType cur_distance = dst_(q_comp, iter.get_point());
            ++num_seen;

            if (initially_inserted < k)
            {
              heap_.insert_unsorted(-cur_distance, iter.get_key());
              ++initially_inserted;
              if (initially_inserted == k)
              {
                heap_.heapify();
              }
            }
            else if (cur_distance < -heap_.min_key())
            {
              record_heap_change(iter);
              heap_.replace_top(-cur_distance, iter.get_key());
            }
            ++iter;
          }

          if (initially_inserted >= k && num_seen >= warmup)
          {
            update_centroid(&q_prime, centroid_is_initialized, k);
            centroid_is_initialized = true;
          }

          old_keys_.clear();
          new_keys_.clear();
        }

        res.resize(initially_inserted);

        // Sorts the heap from start to the kth thing
        std::sort(heap_.get_data().begin(),
                  heap_.get_data().begin() + initially_inserted);
        for (int_fast64_t ii = 0; ii < initially_inserted; ++ii)
        {
          res[ii] = heap_.get_data()[initially_inserted - ii - 1].data;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_distance =
            std::chrono::duration_cast<std::chrono::duration<double>>(
                end_time - distance_start_time);
        auto elapsed_total =
            std::chrono::duration_cast<std::chrono::duration<double>>(
                end_time - start_time);
        stats_.average_distance_time += elapsed_distance.count();
        stats_.average_total_query_time += elapsed_total.count();
      }

      void
      find_near_neighbors(const LSHTablePointType &q,
                          const ComparisonPointType &q_comp,
                          DistanceType threshold, int_fast64_t num_probes,
                          int_fast64_t max_num_candidates,
                          std::vector<LSHTableKeyType> *result)
      {
        if (result == nullptr)
        {
          throw NearestNeighborQueryError("Results vector pointer is nullptr.");
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        std::vector<LSHTableKeyType> &res = *result;
        res.clear();

        table_query_->get_unique_candidates(q, num_probes, max_num_candidates,
                                            &candidates_);
        auto distance_start_time = std::chrono::high_resolution_clock::now();

        typename DataStorage::SubsequenceIterator iter =
            data_storage_.get_subsequence(candidates_);
        while (iter.is_valid())
        {
          DistanceType cur_distance = dst_(q_comp, iter.get_point());
          if (cur_distance < threshold)
          {
            res.push_back(iter.get_key());
          }
          ++iter;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_distance =
            std::chrono::duration_cast<std::chrono::duration<double>>(
                end_time - distance_start_time);
        auto elapsed_total =
            std::chrono::duration_cast<std::chrono::duration<double>>(end_time -
                                                                      start_time);
        stats_.average_distance_time += elapsed_distance.count();
        stats_.average_total_query_time += elapsed_total.count();
      }

      void get_candidates_with_duplicates(const LSHTablePointType &q,
                                          int_fast64_t num_probes,
                                          int_fast64_t max_num_candidates,
                                          std::vector<LSHTableKeyType> *result)
      {
        auto start_time = std::chrono::high_resolution_clock::now();

        table_query_->get_candidates_with_duplicates(q, num_probes,
                                                     max_num_candidates, result);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_total =
            std::chrono::duration_cast<std::chrono::duration<double>>(end_time -
                                                                      start_time);
        stats_.average_total_query_time += elapsed_total.count();
      }

      void get_unique_candidates(const LSHTablePointType &q,
                                 int_fast64_t num_probes,
                                 int_fast64_t max_num_candidates,
                                 std::vector<LSHTableKeyType> *result)
      {
        auto start_time = std::chrono::high_resolution_clock::now();

        table_query_->get_unique_candidates(q, num_probes, max_num_candidates,
                                            result);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_total =
            std::chrono::duration_cast<std::chrono::duration<double>>(end_time -
                                                                      start_time);
        stats_.average_total_query_time += elapsed_total.count();
      }

      void reset_query_statistics()
      {
        table_query_->reset_query_statistics();
        stats_.reset();
      }

      QueryStatistics get_query_statistics()
      {
        QueryStatistics res = table_query_->get_query_statistics();
        res.average_total_query_time = stats_.average_total_query_time;
        res.average_distance_time = stats_.average_distance_time;

        if (res.num_queries > 0)
        {
          res.average_total_query_time /= res.num_queries;
          res.average_distance_time /= res.num_queries;
        }
        return res;
      }

    private:
      LSHTableQuery *table_query_;
      const DataStorage &data_storage_;
      std::vector<LSHTableKeyType> candidates_;
      DistanceFunction dst_;
      SimpleHeap<DistanceType, LSHTableKeyType> heap_;
      std::vector<LSHTableKeyType> old_keys_;
      std::vector<LSHTableKeyType> new_keys_;

      QueryStatistics stats_;
    };

  } // namespace core
} // namespace falconn

#endif
