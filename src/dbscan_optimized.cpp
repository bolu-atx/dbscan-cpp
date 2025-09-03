#include "dbscan_optimized.h"
#include "parallel.hpp"

namespace dbscan {

template <typename T> ClusterResult<T> DBSCANOptimized<T>::cluster(const std::vector<Point<T>> &points) const {
  const int32_t n_points = points.size();
  if (n_points == 0) {
    return {{}, 0};
  }
  const T epsilon_sq = eps_ * eps_;

  struct WorkingPoint {
    T x, y;
    int32_t cluster_id = -1;
    int32_t cell_id = -1;
    bool is_core = false;
  };

  std::vector<WorkingPoint> working_points(n_points);
  for (int32_t i = 0; i < n_points; ++i) {
    working_points[i].x = points[i].x;
    working_points[i].y = points[i].y;
  }

  // Step 1: Grid Indexing
  T min_x = working_points[0].x, max_x = working_points[0].x;
  T min_y = working_points[0].y, max_y = working_points[0].y;
  for (int32_t i = 1; i < n_points; ++i) {
    min_x = std::min(min_x, working_points[i].x);
    max_x = std::max(max_x, working_points[i].x);
    min_y = std::min(min_y, working_points[i].y);
    max_y = std::max(max_y, working_points[i].y);
  }

  const int32_t cells_x = static_cast<int32_t>((max_x - min_x) / eps_) + 1;
  const int32_t cells_y = static_cast<int32_t>((max_y - min_y) / eps_) + 1;
  const int32_t num_cells = cells_x * cells_y;
  std::vector<std::vector<int32_t>> grid(num_cells);

  for (int32_t i = 0; i < n_points; ++i) {
    int32_t cx = static_cast<int32_t>((working_points[i].x - min_x) / eps_);
    int32_t cy = static_cast<int32_t>((working_points[i].y - min_y) / eps_);
    working_points[i].cell_id = cx + cy * cells_x;
  }

  for (int32_t i = 0; i < n_points; ++i) {
    grid[working_points[i].cell_id].push_back(i);
  }

  // Step 2: Core Point Detection (parallel)
  utils::parallel_for(0, n_points, 0, std::function<void(size_t, size_t)>([&](size_t start, size_t end) {
                        for (size_t i = start; i < end; ++i) {
                          int32_t neighbor_count = 0;
                          int32_t cx = working_points[i].cell_id % cells_x;
                          int32_t cy = working_points[i].cell_id / cells_x;

                          for (int32_t dx = -1; dx <= 1; ++dx) {
                            for (int32_t dy = -1; dy <= 1; ++dy) {
                              int32_t neighbor_cx = cx + dx;
                              int32_t neighbor_cy = cy + dy;

                              if (neighbor_cx >= 0 && neighbor_cx < cells_x && neighbor_cy >= 0 &&
                                  neighbor_cy < cells_y) {
                                int32_t neighbor_cell_id = neighbor_cx + neighbor_cy * cells_x;
                                for (int32_t neighbor_idx : grid[neighbor_cell_id]) {
                                  if (neighbor_idx == static_cast<int32_t>(i))
                                    continue;
                                  T dist_sq = distance_squared(points[i], points[neighbor_idx]);
                                  if (dist_sq <= epsilon_sq) {
                                    neighbor_count++;
                                  }
                                }
                              }
                            }
                          }
                          if (neighbor_count >= min_pts_) {
                            working_points[i].is_core = true;
                          }
                        }
                      }));

  // Step 3: Connected Components (parallel)
  AtomicUnionFind uf(n_points);
  utils::parallel_for(0, n_points, nthreads_, std::function<void(size_t, size_t)>([&](size_t start, size_t end) {
                        for (size_t i = start; i < end; ++i) {
                          if (!working_points[i].is_core)
                            continue;
                          int32_t cx = working_points[i].cell_id % cells_x;
                          int32_t cy = working_points[i].cell_id / cells_x;

                          for (int32_t dx = -1; dx <= 1; ++dx) {
                            for (int32_t dy = -1; dy <= 1; ++dy) {
                              int32_t neighbor_cx = cx + dx;
                              int32_t neighbor_cy = cy + dy;
                              if (neighbor_cx >= 0 && neighbor_cx < cells_x && neighbor_cy >= 0 &&
                                  neighbor_cy < cells_y) {
                                int32_t neighbor_cell_id = neighbor_cx + neighbor_cy * cells_x;
                                for (int32_t neighbor_idx : grid[neighbor_cell_id]) {
                                  if (static_cast<int32_t>(i) == neighbor_idx || !working_points[neighbor_idx].is_core)
                                    continue;
                                  T dist_sq = distance_squared(points[i], points[neighbor_idx]);
                                  if (dist_sq <= epsilon_sq) {
                                    uf.unite(i, neighbor_idx);
                                  }
                                }
                              }
                            }
                          }
                        }
                      }));

  // Step 4: Label Clusters (parallel)
  utils::parallel_for(0, n_points, this->nthreads_, std::function<void(size_t, size_t)>([&](size_t start, size_t end) {
                        for (size_t i = start; i < end; ++i) {
                          if (working_points[i].is_core) {
                            working_points[i].cluster_id = uf.find(i);
                          }
                        }
                      }));

  // Step 5: Assign Border Points (parallel)
  utils::parallel_for(0, n_points, this->nthreads_, std::function<void(size_t, size_t)>([&](size_t start, size_t end) {
                        for (size_t i = start; i < end; ++i) {
                          if (working_points[i].is_core)
                            continue;
                          int32_t cx = working_points[i].cell_id % cells_x;
                          int32_t cy = working_points[i].cell_id / cells_x;
                          bool assigned = false;
                          for (int32_t dx = -1; dx <= 1 && !assigned; ++dx) {
                            for (int32_t dy = -1; dy <= 1 && !assigned; ++dy) {
                              int32_t neighbor_cx = cx + dx;
                              int32_t neighbor_cy = cy + dy;
                              if (neighbor_cx >= 0 && neighbor_cx < cells_x && neighbor_cy >= 0 &&
                                  neighbor_cy < cells_y) {
                                int32_t neighbor_cell_id = neighbor_cx + neighbor_cy * cells_x;
                                for (int32_t neighbor_idx : grid[neighbor_cell_id]) {
                                  if (working_points[neighbor_idx].is_core) {
                                    T dist_sq = distance_squared(points[i], points[neighbor_idx]);
                                    if (dist_sq <= epsilon_sq) {
                                      working_points[i].cluster_id = working_points[neighbor_idx].cluster_id;
                                      assigned = true;
                                      break;
                                    }
                                  }
                                }
                              }
                            }
                          }
                        }
                      }));

  // Step 6: Finalize and Return Result
  std::vector<int32_t> labels(n_points);
  std::unordered_set<int> cluster_ids;
  for (int32_t i = 0; i < n_points; ++i) {
    labels[i] = working_points[i].cluster_id;
    if (labels[i] != -1) {
      cluster_ids.insert(static_cast<int>(labels[i]));
    }
  }

  return {labels, static_cast<int32_t>(cluster_ids.size())};
}

// Explicit template instantiations
template class DBSCANOptimized<double>;
template class DBSCANOptimized<float>;

} // namespace dbscan