#pragma once

#include "dbscan_grid2d_l1.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <numeric>
#include <utility>
#include <vector>

#include "parallel.hpp"
#include "perf_timer.h"

namespace dbscan::detail {

constexpr uint64_t pack_cell(uint32_t ix, uint32_t iy) noexcept {
  return (static_cast<uint64_t>(ix) << 32U) | static_cast<uint64_t>(iy);
}

constexpr uint32_t cell_of(uint32_t value, uint32_t cell_size) noexcept {
  return cell_size == 0 ? value : static_cast<uint32_t>(value / cell_size);
}

inline uint32_t load_coord(const uint32_t *ptr, std::size_t stride, std::size_t index) noexcept {
  return *(ptr + index * stride);
}

template <typename Fn>
void for_each_neighbor(uint32_t point_index, uint32_t eps, const uint32_t *x, std::size_t x_stride,
                       const uint32_t *y, std::size_t y_stride, const std::vector<uint32_t> &cell_x,
                       const std::vector<uint32_t> &cell_y, const std::vector<uint32_t> &ordered_indices,
                       const std::vector<std::size_t> &cell_offsets, const std::vector<uint64_t> &unique_keys,
                       Fn &&fn) {

  const uint32_t base_cx = cell_x[point_index];
  const uint32_t base_cy = cell_y[point_index];
  const uint32_t x_a = load_coord(x, x_stride, point_index);
  const uint32_t y_a = load_coord(y, y_stride, point_index);

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

        const uint32_t x_b = load_coord(x, x_stride, neighbor_idx);
        const uint32_t y_b = load_coord(y, y_stride, neighbor_idx);

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
  std::size_t x_stride;
  const uint32_t *y;
  std::size_t y_stride;
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
      for_each_neighbor(current, ctx.eps, ctx.x, ctx.x_stride, ctx.y, ctx.y_stride, ctx.cell_x, ctx.cell_y,
                        ctx.ordered_indices, ctx.cell_offsets, ctx.unique_keys, [&](uint32_t neighbor) {
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
          for_each_neighbor(current, ctx.eps, ctx.x, ctx.x_stride, ctx.y, ctx.y_stride, ctx.cell_x, ctx.cell_y,
                            ctx.ordered_indices, ctx.cell_offsets, ctx.unique_keys, [&](uint32_t neighbor) {
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

      for_each_neighbor(static_cast<uint32_t>(idx), ctx.eps, ctx.x, ctx.x_stride, ctx.y, ctx.y_stride, ctx.cell_x,
                        ctx.cell_y, ctx.ordered_indices, ctx.cell_offsets, ctx.unique_keys, [&](uint32_t neighbor) {
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
    for_each_neighbor(static_cast<uint32_t>(i), ctx.eps, ctx.x, ctx.x_stride, ctx.y, ctx.y_stride, ctx.cell_x,
                      ctx.cell_y, ctx.ordered_indices, ctx.cell_offsets, ctx.unique_keys, [&](uint32_t neighbor) {
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

inline DBSCANGrid2DL1Result dbscan_grid2d_l1_impl(const uint32_t *x, std::size_t x_stride, const uint32_t *y,
                                                  std::size_t y_stride, std::size_t count,
                                                  const DBSCANGrid2DL1Params &params,
                                                  GridExpansionMode expansion_mode) {
  DBSCANGrid2DL1Result result;
  result.perf_timing.clear();
  if (count == 0)
    return result;

  ScopedTimer total_timer("total", result.perf_timing);

  const uint32_t cell_size = params.eps;

  std::vector<uint32_t> cell_x(count);
  std::vector<uint32_t> cell_y(count);
  std::vector<uint64_t> keys(count);
  std::vector<uint32_t> ordered_indices(count);
  std::iota(ordered_indices.begin(), ordered_indices.end(), 0U);

  const std::size_t index_chunk = params.chunk_size == 0 ? 1024 : params.chunk_size;
  {
    ScopedTimer timer("precompute_cells", result.perf_timing);
    utils::parallelize(0, count, params.num_threads, index_chunk, [&](std::size_t begin, std::size_t end) {
      for (std::size_t i = begin; i < end; ++i) {
        const uint32_t cx = cell_of(load_coord(x, x_stride, i), cell_size);
        const uint32_t cy = cell_of(load_coord(y, y_stride, i), cell_size);
        cell_x[i] = cx;
        cell_y[i] = cy;
        keys[i] = pack_cell(cx, cy);
      }
    });
  }

  {
    ScopedTimer timer("sort_indices", result.perf_timing);
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
    ScopedTimer timer("build_cell_offsets", result.perf_timing);
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

  result.labels.assign(count, -1);
  std::vector<uint8_t> is_core(count, 0U);

  const uint32_t eps_value = params.eps;
  const uint32_t min_samples_value = params.min_samples;

  const std::size_t core_chunk = params.chunk_size == 0 ? 512 : params.chunk_size;
  {
    ScopedTimer timer("core_detection", result.perf_timing);
    utils::parallelize(0, count, params.num_threads, core_chunk, [&](std::size_t begin, std::size_t end) {
      for (std::size_t idx = begin; idx < end; ++idx) {
        uint32_t neighbor_count = 0;
        for_each_neighbor(static_cast<uint32_t>(idx), eps_value, x, x_stride, y, y_stride, cell_x, cell_y,
                          ordered_indices, cell_offsets, unique_keys, [&](uint32_t) {
                            ++neighbor_count;
                            return neighbor_count < min_samples_value;
                          });

        if (neighbor_count >= min_samples_value)
          is_core[idx] = 1U;
      }
    });
  }

  const ExpansionContext context{x,
                                 x_stride,
                                 y,
                                 y_stride,
                                 count,
                                 eps_value,
                                 min_samples_value,
                                 cell_x,
                                 cell_y,
                                 ordered_indices,
                                 cell_offsets,
                                 unique_keys,
                                 is_core,
                                 params.num_threads,
                                 params.chunk_size};

  {
    ScopedTimer timer("cluster_expansion", result.perf_timing);
    switch (expansion_mode) {
    case GridExpansionMode::Sequential:
      sequential_expand(context, result.labels);
      break;
    case GridExpansionMode::FrontierParallel:
      frontier_expand(context, result.labels);
      break;
    case GridExpansionMode::UnionFind:
      union_find_expand(context, result.labels);
      break;
    }
  }

  return result;
}

} // namespace dbscan::detail

