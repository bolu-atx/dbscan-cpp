#include "dbscan_grid2d_l1.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <mutex>
#include <utility>
#include <vector>

#include "parallel.hpp"
#include "perf_timer.h"

namespace dbscan {

namespace {

// Compact 2D cell coordinates into a sortable key so we can reuse std::sort rather than bespoke grid maps.
constexpr uint64_t pack_cell(uint32_t ix, uint32_t iy) noexcept {
  return (static_cast<uint64_t>(ix) << 32U) | static_cast<uint64_t>(iy);
}

constexpr uint32_t cell_of(uint32_t value, uint32_t cell_size) noexcept {
  return cell_size == 0 ? value : static_cast<uint32_t>(value / cell_size);
}

// Neighbors are explored by scanning adjacent grid cells so that the expensive L1 radius test only touches
// candidates that share the precomputed bucket, keeping the branch predictor in our favor.
template <typename Fn>
void for_each_neighbor(uint32_t point_index, uint32_t eps, const uint32_t *x, const uint32_t *y,
                       const std::vector<uint32_t> &cell_x, const std::vector<uint32_t> &cell_y,
                       const std::vector<uint32_t> &ordered_indices, const std::vector<size_t> &cell_offsets,
                       const std::vector<uint64_t> &unique_keys, Fn &&fn) {

  const uint32_t base_cx = cell_x[point_index];
  const uint32_t base_cy = cell_y[point_index];

  for (int dx = -1; dx <= 1; ++dx) {
    const int64_t nx = static_cast<int64_t>(base_cx) + dx;
    if (nx < 0)
      continue;

    for (int dy = -1; dy <= 1; ++dy) {
      const int64_t ny = static_cast<int64_t>(base_cy) + dy;
      if (ny < 0)
        continue;

      const uint64_t key = pack_cell(static_cast<uint32_t>(nx), static_cast<uint32_t>(ny));
      const auto it = std::lower_bound(unique_keys.begin(), unique_keys.end(), key);
      if (it == unique_keys.end() || *it != key)
        continue;

      const std::size_t cell_idx = static_cast<std::size_t>(std::distance(unique_keys.begin(), it));
      const std::size_t begin = cell_offsets[cell_idx];
      const std::size_t end = cell_offsets[cell_idx + 1];

      for (std::size_t pos = begin; pos < end; ++pos) {
        const uint32_t neighbor_idx = ordered_indices[pos];

        const uint32_t x_a = x[point_index];
        const uint32_t y_a = y[point_index];
        const uint32_t x_b = x[neighbor_idx];
        const uint32_t y_b = y[neighbor_idx];

        const uint32_t dx_abs = x_a > x_b ? x_a - x_b : x_b - x_a;
        const uint32_t dy_abs = y_a > y_b ? y_a - y_b : y_b - y_a;
        const uint64_t manhattan = static_cast<uint64_t>(dx_abs) + static_cast<uint64_t>(dy_abs);

        if (manhattan <= static_cast<uint64_t>(eps)) {
          if (!fn(neighbor_idx))
            return;
        }
      }
    }
  }
}

struct ExpansionContext {
  const uint32_t *x;
  const uint32_t *y;
  std::size_t count;
  uint32_t eps;
  uint32_t min_samples;
  const std::vector<uint32_t> &cell_x;
  const std::vector<uint32_t> &cell_y;
  const std::vector<uint32_t> &ordered_indices;
  const std::vector<std::size_t> &cell_offsets;
  const std::vector<uint64_t> &unique_keys;
  const std::vector<uint8_t> &is_core;
  std::size_t num_threads;
  std::size_t chunk_size;
};

void sequential_expand(const ExpansionContext &ctx, std::vector<int32_t> &labels) {
  std::vector<uint32_t> stack;
  stack.reserve(ctx.count);
  std::vector<uint32_t> neighbor_buffer;
  neighbor_buffer.reserve(64);

  int32_t next_label = 0;
  for (std::size_t i = 0; i < ctx.count; ++i) {
    if (!ctx.is_core[i] || labels[i] != -1)
      continue;

    labels[i] = next_label;
    stack.clear();
    stack.push_back(static_cast<uint32_t>(i));

    while (!stack.empty()) {
      const uint32_t current = stack.back();
      stack.pop_back();

      neighbor_buffer.clear();
      // Capture neighbors into a buffer first so every touch of labels happens after the search, keeping the
      // expansion phase deterministic regardless of how for_each_neighbor yields matches.
      for_each_neighbor(current, ctx.eps, ctx.x, ctx.y, ctx.cell_x, ctx.cell_y, ctx.ordered_indices, ctx.cell_offsets,
                        ctx.unique_keys, [&](uint32_t neighbor) {
                          neighbor_buffer.push_back(neighbor);
                          return true;
                        });

      for (uint32_t neighbor : neighbor_buffer) {
        if (labels[neighbor] == -1) {
          labels[neighbor] = next_label;
          if (ctx.is_core[neighbor])
            stack.push_back(neighbor);
        }
      }
    }

    ++next_label;
  }
}

void frontier_expand(const ExpansionContext &ctx, std::vector<int32_t> &labels);

void union_find_expand(const ExpansionContext &ctx, std::vector<int32_t> &labels);

void frontier_expand(const ExpansionContext &ctx, std::vector<int32_t> &labels) {
  std::vector<std::atomic<int32_t>> shared_labels(ctx.count);
  for (std::size_t i = 0; i < ctx.count; ++i)
    shared_labels[i].store(labels[i], std::memory_order_relaxed);

  int32_t next_label = 0;
  std::vector<uint32_t> frontier;
  frontier.reserve(256);

  const std::size_t frontier_chunk = ctx.chunk_size == 0 ? 64 : ctx.chunk_size;

  for (std::size_t seed = 0; seed < ctx.count; ++seed) {
    if (!ctx.is_core[seed] || shared_labels[seed].load(std::memory_order_acquire) != -1)
      continue;

    const int32_t label = next_label++;
    shared_labels[seed].store(label, std::memory_order_release);
    frontier.clear();
    frontier.push_back(static_cast<uint32_t>(seed));

    while (!frontier.empty()) {
      std::vector<uint32_t> next_frontier;
      std::mutex next_mutex;

      utils::parallelize(0, frontier.size(), ctx.num_threads, frontier_chunk, [&](std::size_t begin, std::size_t end) {
        std::vector<uint32_t> local_next;
        local_next.reserve(32);
        std::vector<uint32_t> neighbor_buffer;
        neighbor_buffer.reserve(64);

        for (std::size_t idx = begin; idx < end; ++idx) {
          const uint32_t current = frontier[idx];

          neighbor_buffer.clear();
          for_each_neighbor(current, ctx.eps, ctx.x, ctx.y, ctx.cell_x, ctx.cell_y, ctx.ordered_indices,
                            ctx.cell_offsets, ctx.unique_keys, [&](uint32_t neighbor) {
                              neighbor_buffer.push_back(neighbor);
                              return true;
                            });

          for (uint32_t neighbor : neighbor_buffer) {
            if (ctx.is_core[neighbor]) {
              int32_t expected = -1;
              if (shared_labels[neighbor].compare_exchange_strong(expected, label, std::memory_order_acq_rel))
                local_next.push_back(neighbor);
            } else {
              int32_t expected = -1;
              shared_labels[neighbor].compare_exchange_strong(expected, label, std::memory_order_acq_rel);
            }
          }
        }

        if (!local_next.empty()) {
          std::sort(local_next.begin(), local_next.end());
          local_next.erase(std::unique(local_next.begin(), local_next.end()), local_next.end());
          std::lock_guard<std::mutex> lock(next_mutex);
          next_frontier.insert(next_frontier.end(), local_next.begin(), local_next.end());
        }
      });

      if (next_frontier.empty())
        break;

      std::sort(next_frontier.begin(), next_frontier.end());
      next_frontier.erase(std::unique(next_frontier.begin(), next_frontier.end()), next_frontier.end());
      frontier.swap(next_frontier);
    }
  }

  for (std::size_t i = 0; i < ctx.count; ++i)
    labels[i] = shared_labels[i].load(std::memory_order_relaxed);
}

void union_find_expand(const ExpansionContext &ctx, std::vector<int32_t> &labels) {
  constexpr uint32_t invalid = std::numeric_limits<uint32_t>::max();
  std::vector<std::atomic<uint32_t>> parents(ctx.count);
  for (std::size_t i = 0; i < ctx.count; ++i) {
    if (ctx.is_core[i])
      parents[i].store(static_cast<uint32_t>(i), std::memory_order_relaxed);
    else
      parents[i].store(invalid, std::memory_order_relaxed);
  }

  auto find_root = [&](uint32_t node) {
    uint32_t parent = parents[node].load(std::memory_order_acquire);
    if (parent == invalid)
      return invalid;

    while (true) {
      uint32_t grandparent = parents[parent].load(std::memory_order_acquire);
      if (grandparent == parent) {
        if (parent != node)
          parents[node].store(parent, std::memory_order_release);
        return parent;
      }
      parents[node].compare_exchange_strong(parent, grandparent, std::memory_order_acq_rel);
      node = parent;
      parent = parents[node].load(std::memory_order_acquire);
      if (parent == invalid)
        return invalid;
    }
  };

  auto unite = [&](uint32_t a, uint32_t b) {
    while (true) {
      a = find_root(a);
      b = find_root(b);
      if (a == invalid || b == invalid || a == b)
        return;

      if (a < b) {
        uint32_t expected = b;
        if (parents[b].compare_exchange_strong(expected, a, std::memory_order_acq_rel))
          return;
      } else {
        uint32_t expected = a;
        if (parents[a].compare_exchange_strong(expected, b, std::memory_order_acq_rel))
          return;
      }
    }
  };

  const std::size_t union_chunk = ctx.chunk_size == 0 ? 512 : ctx.chunk_size;
  utils::parallelize(0, ctx.count, ctx.num_threads, union_chunk, [&](std::size_t begin, std::size_t end) {
    for (std::size_t idx = begin; idx < end; ++idx) {
      if (!ctx.is_core[idx])
        continue;

      for_each_neighbor(static_cast<uint32_t>(idx), ctx.eps, ctx.x, ctx.y, ctx.cell_x, ctx.cell_y, ctx.ordered_indices,
                        ctx.cell_offsets, ctx.unique_keys, [&](uint32_t neighbor) {
                          if (ctx.is_core[neighbor])
                            unite(static_cast<uint32_t>(idx), neighbor);
                          return true;
                        });
    }
  });

  std::vector<uint32_t> root_for_point(ctx.count, invalid);
  for (std::size_t i = 0; i < ctx.count; ++i) {
    if (ctx.is_core[i])
      root_for_point[i] = find_root(static_cast<uint32_t>(i));
  }

  std::vector<uint32_t> component_min(ctx.count, invalid);
  for (std::size_t i = 0; i < ctx.count; ++i) {
    if (!ctx.is_core[i])
      continue;
    const uint32_t root = root_for_point[i];
    if (root == invalid)
      continue;
    if (component_min[root] > i)
      component_min[root] = static_cast<uint32_t>(i);
  }

  std::vector<std::pair<uint32_t, uint32_t>> components;
  components.reserve(ctx.count);
  for (std::size_t i = 0; i < ctx.count; ++i) {
    if (component_min[i] != invalid)
      components.emplace_back(component_min[i], static_cast<uint32_t>(i));
  }

  std::sort(components.begin(), components.end());

  std::vector<int32_t> root_label(ctx.count, -1);
  int32_t next_label = 0;
  for (const auto &[min_index, root] : components) {
    (void)min_index;
    root_label[root] = next_label++;
  }

  for (std::size_t i = 0; i < ctx.count; ++i) {
    if (!ctx.is_core[i])
      continue;
    const uint32_t root = root_for_point[i];
    if (root == invalid)
      continue;
    labels[i] = root_label[root];
  }

  for (std::size_t i = 0; i < ctx.count; ++i) {
    if (ctx.is_core[i])
      continue;

    int32_t best_label = -1;
    for_each_neighbor(static_cast<uint32_t>(i), ctx.eps, ctx.x, ctx.y, ctx.cell_x, ctx.cell_y, ctx.ordered_indices,
                      ctx.cell_offsets, ctx.unique_keys, [&](uint32_t neighbor) {
                        if (!ctx.is_core[neighbor])
                          return true;
                        const int32_t candidate = labels[neighbor];
                        if (candidate != -1 && (best_label == -1 || candidate < best_label))
                          best_label = candidate;
                        return true;
                      });
    labels[i] = best_label;
  }
}
} // namespace

DBSCANGrid2D_L1::DBSCANGrid2D_L1(uint32_t eps_value, uint32_t min_samples_value, std::size_t num_threads_value,
                                 std::size_t chunk_size_value, GridExpansionMode mode_value)
    : eps(eps_value), min_samples(min_samples_value), num_threads(num_threads_value), chunk_size(chunk_size_value),
      expansion_mode(mode_value) {
  if (eps == 0)
    throw std::invalid_argument("eps must be greater than zero for DBSCANGrid2D_L1");
  if (min_samples == 0)
    throw std::invalid_argument("min_samples must be greater than zero for DBSCANGrid2D_L1");
}

std::vector<int32_t> DBSCANGrid2D_L1::fit_predict(const uint32_t *x, const uint32_t *y, std::size_t count) {
  if (x == nullptr || y == nullptr)
    throw std::invalid_argument("Input coordinate arrays must be non-null");

  if (count == 0)
    return {};

  perf_timing_.clear();
  ScopedTimer total_timer("total", perf_timing_);

  const uint32_t cell_size = eps;

  std::vector<uint32_t> cell_x(count);
  std::vector<uint32_t> cell_y(count);
  std::vector<uint64_t> keys(count);
  std::vector<uint32_t> ordered_indices(count);
  std::iota(ordered_indices.begin(), ordered_indices.end(), 0U);

  const std::size_t index_chunk = chunk_size == 0 ? 1024 : chunk_size;
  {
    // Precompute grid placements in parallel so later stages can stay read-only and avoid rehashing coordinates.
    ScopedTimer timer("precompute_cells", perf_timing_);
    utils::parallelize(0, count, num_threads, index_chunk, [&](std::size_t begin, std::size_t end) {
      for (std::size_t i = begin; i < end; ++i) {
        const uint32_t cx = cell_of(x[i], cell_size);
        const uint32_t cy = cell_of(y[i], cell_size);
        cell_x[i] = cx;
        cell_y[i] = cy;
        keys[i] = pack_cell(cx, cy);
      }
    });
  }

  {
    // Sorting indices by packed cell ensures neighbors form contiguous spans which we can scan without lookups.
    ScopedTimer timer("sort_indices", perf_timing_);
    std::sort(ordered_indices.begin(), ordered_indices.end(), [&](uint32_t lhs, uint32_t rhs) {
      const uint64_t key_lhs = keys[lhs];
      const uint64_t key_rhs = keys[rhs];
      if (key_lhs == key_rhs)
        return lhs < rhs;
      return key_lhs < key_rhs;
    });
  }

  std::vector<std::size_t> cell_offsets;
  cell_offsets.reserve(count + 1);
  std::vector<uint64_t> unique_keys;
  unique_keys.reserve(count);

  {
    // Build a CSR-style view of the sorted cells so we can jump directly to the occupants of any neighboring bucket.
    ScopedTimer timer("build_cell_offsets", perf_timing_);
    std::size_t pos = 0;
    while (pos < count) {
      const uint64_t key = keys[ordered_indices[pos]];
      unique_keys.push_back(key);
      cell_offsets.push_back(pos);

      do {
        ++pos;
      } while (pos < count && keys[ordered_indices[pos]] == key);
    }
    cell_offsets.push_back(count);
  }

  std::vector<int32_t> labels(count, -1);
  std::vector<uint8_t> is_core(count, 0U);

  const uint32_t eps_value = eps;
  const uint32_t min_samples_value = min_samples;

  const std::size_t core_chunk = chunk_size == 0 ? 512 : chunk_size;
  {
    // Core detection runs as an isolated pass so expansion can treat label writes as the only mutation, simplifying
    // synchronization even when the search function is invoked concurrently.
    ScopedTimer timer("core_detection", perf_timing_);
    utils::parallelize(0, count, num_threads, core_chunk, [&](std::size_t begin, std::size_t end) {
      for (std::size_t idx = begin; idx < end; ++idx) {
        uint32_t neighbor_count = 0;
        for_each_neighbor(static_cast<uint32_t>(idx), eps_value, x, y, cell_x, cell_y, ordered_indices, cell_offsets,
                          unique_keys, [&](uint32_t) {
                            ++neighbor_count;
                            return neighbor_count < min_samples_value;
                          });

        if (neighbor_count >= min_samples_value)
          is_core[idx] = 1U;
      }
    });
  }

  const ExpansionContext context{x,
                                 y,
                                 count,
                                 eps_value,
                                 min_samples_value,
                                 cell_x,
                                 cell_y,
                                 ordered_indices,
                                 cell_offsets,
                                 unique_keys,
                                 is_core,
                                 num_threads,
                                 chunk_size};

  {
    ScopedTimer timer("cluster_expansion", perf_timing_);
    switch (expansion_mode) {
    case GridExpansionMode::Sequential:
      sequential_expand(context, labels);
      break;
    case GridExpansionMode::FrontierParallel:
      frontier_expand(context, labels);
      break;
    case GridExpansionMode::UnionFind:
      union_find_expand(context, labels);
      break;
    }
  }

  return labels;
}

} // namespace dbscan
