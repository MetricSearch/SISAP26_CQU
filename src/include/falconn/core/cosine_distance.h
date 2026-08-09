#ifndef __COSINE_DISTANCE_H__
#define __COSINE_DISTANCE_H__

#include <cstdint>
#include <vector>

#include <Eigen/Dense>

namespace falconn {
namespace core {

// TODO: rename to negative inner product distance?
// TODO: make a single CosineDistance class with different template
// specializations?

// The Sparse functions assume that the data points are stored as a
// std::vector of (index, coefficient) pairs. The indices must be sorted.

template <typename CoordinateType = float, typename IndexType = int32_t>
struct CosineDistanceSparse {
  typedef std::vector<std::pair<IndexType, CoordinateType>> VectorType;

  CoordinateType operator()(const VectorType& p1, const VectorType& p2) {
    size_t i = 0, j = 0;
    float sum = 0.0f;

    while (i < p1.size() && j < p2.size()) {
        const int ai = p1[i].first;
        const int bj = p2[j].first;

        if (ai == bj) {
            sum += p1[i].second * p2[j].second;
            ++i;
            ++j;
        } else if (ai < bj) {
            ++i;
        } else {
            ++j;
        }
    }

    return -sum;
  }
};

// The Dense functions assume that the data points are stored as dense
// Eigen column vectors.

template <typename CoordinateType = float>
struct CosineDistanceDense {
  typedef Eigen::Matrix<CoordinateType, Eigen::Dynamic, 1, Eigen::ColMajor>
      VectorType;

  template <typename Derived1, typename Derived2>
  CoordinateType operator()(const Eigen::MatrixBase<Derived1>& p1,
                            const Eigen::MatrixBase<Derived2>& p2) {
    // negate the result because LSHTable assumes that smaller distances
    // are better
    return -p1.dot(p2);
  }
};

}  // namespace core
}  // namespace falconn

#endif
