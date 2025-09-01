#pragma once

#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>

namespace dbscan {

template <typename T = double> struct Point {
  T x, y;
};

template <typename T = double> struct ClusterResult {
  std::vector<int32_t> labels; // -1 for noise, cluster id for core/border points
  int32_t num_clusters;
};

template <typename T = double> class DBSCAN {
public:
  /**
   * @brief Constructs a DBSCAN clustering algorithm instance.
   * @param eps Maximum distance between two points for them to be considered neighbors.
   * @param min_pts Minimum number of points required to form a dense region (core point).
   */
  DBSCAN(T eps, int32_t min_pts);

  /**
   * @brief Performs DBSCAN clustering on the given set of points.
   * @param points Vector of 2D points to cluster.
   * @return ClusterResult containing cluster labels and number of clusters found.
   */
  ClusterResult<T> cluster(const std::vector<Point<T>> &points) const;

private:
  T eps_;
  int32_t min_pts_;

  // Helper functions
protected:
  std::vector<int32_t> find_neighbors(const std::vector<Point<T>> &points, int32_t point_idx) const;

  /**
   * @brief Computes squared Euclidean distance between two points (inlined for performance).
   * @param a First point.
   * @param b Second point.
   * @return Squared distance between points.
   */
  inline T distance_squared(const Point<T> &a, const Point<T> &b) const {
    T dx = a.x - b.x;
    T dy = a.y - b.y;
    return dx * dx + dy * dy;
  }

  void expand_cluster(const std::vector<Point<T>> &points, std::vector<int32_t> &labels, int32_t point_idx,
                      int32_t cluster_id, const std::vector<int32_t> &neighbors) const;
};

} // namespace dbscan