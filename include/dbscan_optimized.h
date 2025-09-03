#pragma once

#include "dbscan.h"
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <unordered_set>
#include <vector>

namespace dbscan {

class AtomicUnionFind {
public:
  explicit AtomicUnionFind(int32_t n) : parent(n) {
    // Initialize each element to be its own parent.
    for (int32_t i = 0; i < n; ++i) {
      parent[i].store(i, std::memory_order_relaxed);
    }
  }

  /**
   * Finds the representative (root) of the set containing element i.
   * Applies path compression along the way.
   */
  int32_t find(int32_t i) {
    // 1. Find the root of the set.
    int32_t root = i;
    while (true) {
      int32_t parent_val = parent[root].load(std::memory_order_relaxed);
      if (parent_val == root) {
        break;
      }
      root = parent_val;
    }

    // 2. Perform path compression.
    int32_t curr = i;
    while (curr != root) {
      int32_t parent_val = parent[curr].load(std::memory_order_relaxed);
      // Atomically update the parent to point to the root.
      // If this fails, another thread has already updated it, which is fine.
      // We don't overwrite a potentially "better" parent with our `root`.
      parent[curr].compare_exchange_weak(parent_val, root, std::memory_order_release, std::memory_order_relaxed);
      curr = parent_val;
    }
    return root;
  }

  /**
   * Unites the sets containing elements i and j.
   */
  void unite(int32_t i, int32_t j) {
    while (true) {
      int32_t root_i = find(i);
      int32_t root_j = find(j);

      if (root_i == root_j) {
        return; // Already in the same set.
      }

      // Always link the smaller root to the larger root for determinism
      // and to help prevent long chains.
      int32_t old_root = std::min(root_i, root_j);
      int32_t new_root = std::max(root_i, root_j);

      int32_t expected = old_root;
      if (parent[old_root].compare_exchange_strong(expected, new_root, std::memory_order_acq_rel)) {
        return; // Success.
      }
      // If CAS failed, another thread interfered. Retry the whole operation.
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
  int32_t nthreads_{0};

  inline T distance_squared(const Point<T> &a, const Point<T> &b) const {
    T dx = a.x - b.x;
    T dy = a.y - b.y;
    return dx * dx + dy * dy;
  }
};

} // namespace dbscan