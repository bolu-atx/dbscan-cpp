#pragma once

#include "dbscan.h"
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <unordered_set>
#include <vector>

namespace dbscan {

// A thread-safe Union-Find data structure using int32_t
class ConcurrentUnionFind {
public:
  ConcurrentUnionFind(int32_t n) : parent(n) {
    for (int32_t i = 0; i < n; ++i) {
      parent[i].store(i);
    }
  }

  int32_t find(int32_t i) {
    int32_t root = i;
    while (root != parent[root].load()) {
      root = parent[root].load();
    }
    int32_t curr = i;
    while (curr != root) {
      int32_t next = parent[curr].load();
      parent[curr].store(root);
      curr = next;
    }
    return root;
  }

  void unite(int32_t i, int32_t j) {
    int32_t root_i = find(i);
    int32_t root_j = find(j);
    if (root_i != root_j) {
      int32_t old_root = std::min(root_i, root_j);
      int32_t new_root = std::max(root_i, root_j);
      parent[old_root].store(new_root);
    }
  }

private:
  std::vector<std::atomic<int32_t>> parent;
};

template <typename T = double> class DBSCANOptimized {
public:
  DBSCANOptimized(T eps, int32_t min_pts) : eps_(eps), min_pts_(min_pts) {}

  ClusterResult<T> cluster(const std::vector<Point<T>> &points) const;

private:
  T eps_;
  int32_t min_pts_;

  inline T distance_squared(const Point<T> &a, const Point<T> &b) const {
    T dx = a.x - b.x;
    T dy = a.y - b.y;
    return dx * dx + dy * dy;
  }
};

} // namespace dbscan