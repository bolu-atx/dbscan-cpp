#include "dbscan_optimized.h"
#include <algorithm>

namespace dbscan {

template <typename T> ClusterResult<T> DBSCANOptimized<T>::cluster() {
  if (points_.empty()) {
    return {{}, 0};
  }

  // Step 1: Find core points in parallel
  std::vector<bool> is_core = find_core_points();

  // Step 2: Process core-core connections using union-find
  UnionFind<T> uf(points_.size());
  process_core_core_connections(is_core, uf);

  // Step 3: Assign border points
  std::vector<int32_t> labels = assign_border_points(is_core, uf);

  // Step 4: Count clusters
  int32_t num_clusters = count_clusters(uf);

  return {labels, num_clusters};
}

template <typename T> std::vector<bool> DBSCANOptimized<T>::find_core_points() const {
  std::vector<bool> is_core(points_.size(), false);

  // TODO: Replace sequential processing with parallel execution using std::execution::par
  // TODO: Consider SIMD vectorization for neighbor counting
  std::for_each(points_.begin(), points_.end(), [&](const Point<T> &point) {
    size_t idx = &point - &points_[0];
    auto neighbors = get_neighbors(idx);
    if (static_cast<int32_t>(neighbors.size()) >= min_pts_) {
      is_core[idx] = true;
    }
  });

  return is_core;
}

template <typename T> std::vector<size_t> DBSCANOptimized<T>::get_neighbors(size_t point_idx) const {
  std::vector<size_t> neighbors;
  const Point<T> &target = points_[point_idx];
  T eps_squared = eps_ * eps_;

  // Get cell coordinates for the target point
  std::pair<size_t, size_t> cell_coords = grid_.get_cell_coords(target);
  size_t cell_x = cell_coords.first;
  size_t cell_y = cell_coords.second;

  // Check neighboring cells
  auto neighbor_cells = grid_.get_neighbor_cells(cell_x, cell_y);

  // TODO: Reserve vector capacity based on estimated neighbor count to avoid reallocations
  // TODO: Consider early termination when min_pts neighbors are found (for core point detection)

  for (auto &cell_coords : neighbor_cells) {
    size_t cx = cell_coords.first;
    size_t cy = cell_coords.second;

    std::vector<size_t> cell_points = grid_.get_points_in_cell(cx, cy);

    for (size_t neighbor_idx : cell_points) {
      if (neighbor_idx == point_idx)
        continue;

      T dist_sq = distance_squared(target, points_[neighbor_idx]);
      if (dist_sq <= eps_squared) {
        neighbors.push_back(neighbor_idx);
      }
    }
  }

  return neighbors;
}

template <typename T>
void DBSCANOptimized<T>::process_core_core_connections(const std::vector<bool> &is_core, UnionFind<T> &uf) const {
  // TODO: Replace sequential processing with parallel union-find operations
  // TODO: Consider path compression optimization in UnionFind::find()
  // TODO: Batch union operations to reduce locking overhead in concurrent scenarios
  std::for_each(points_.begin(), points_.end(), [&](const Point<T> &point) {
    size_t idx = &point - &points_[0];
    if (!is_core[idx])
      return;

    auto neighbors = get_neighbors(idx);
    for (size_t neighbor_idx : neighbors) {
      if (is_core[neighbor_idx] && neighbor_idx > idx) {
        uf.union_sets(static_cast<int32_t>(idx), static_cast<int32_t>(neighbor_idx));
      }
    }
  });
}

template <typename T>
std::vector<int32_t> DBSCANOptimized<T>::assign_border_points(const std::vector<bool> &is_core,
                                                              UnionFind<T> &uf) const {
  std::vector<int32_t> labels(points_.size(), -1);

  // Sequential border point assignment
  std::for_each(points_.begin(), points_.end(), [&](const Point<T> &point) {
    size_t idx = &point - &points_[0];

    if (is_core[idx]) {
      // Core points get their cluster ID
      labels[idx] = uf.find(static_cast<int32_t>(idx));
    } else {
      // Border points: find nearest core point's cluster
      auto neighbors = get_neighbors(idx);
      for (size_t neighbor_idx : neighbors) {
        if (is_core[neighbor_idx]) {
          labels[idx] = uf.find(static_cast<int32_t>(neighbor_idx));
          break; // Take first core neighbor found
        }
      }
    }
  });

  return labels;
}

template <typename T> int32_t DBSCANOptimized<T>::count_clusters(UnionFind<T> &uf) const {
  std::unordered_set<int32_t> unique_clusters;

  // TODO: Optimize cluster counting - could use a vector<bool> or bitset for dense cluster IDs
  // TODO: Consider parallel counting with atomic operations for very large datasets
  for (size_t i = 0; i < points_.size(); ++i) {
    int32_t cluster_id = uf.find(static_cast<int32_t>(i));
    if (cluster_id >= 0) { // Only count non-noise points
      unique_clusters.insert(cluster_id);
    }
  }

  return static_cast<int32_t>(unique_clusters.size());
}

// Explicit template instantiations
template class DBSCANOptimized<double>;
template class DBSCANOptimized<float>;

} // namespace dbscan