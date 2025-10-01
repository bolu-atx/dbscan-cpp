#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "perf_timer.h"

namespace dbscan {

enum class GridExpansionMode {
  Sequential,
  FrontierParallel,
  UnionFind
};

struct DBSCANGrid2D_L1 {
  uint32_t eps;
  uint32_t min_samples;
  std::size_t num_threads;
  std::size_t chunk_size;
  GridExpansionMode expansion_mode;

  DBSCANGrid2D_L1(uint32_t eps_value, uint32_t min_samples_value, std::size_t num_threads_value = 0,
                  std::size_t chunk_size_value = 0, GridExpansionMode mode_value = GridExpansionMode::Sequential);

  [[nodiscard]] std::vector<int32_t> fit_predict(const uint32_t *x, const uint32_t *y, std::size_t count);

  PerfTiming perf_timing_;
};

template <GridExpansionMode Mode>
struct DBSCANGrid2D_L1T : DBSCANGrid2D_L1 {
  DBSCANGrid2D_L1T(uint32_t eps_value, uint32_t min_samples_value, std::size_t num_threads_value = 0,
                   std::size_t chunk_size_value = 0)
      : DBSCANGrid2D_L1(eps_value, min_samples_value, num_threads_value, chunk_size_value, Mode) {}
};

using DBSCANGrid2D_L1Frontier = DBSCANGrid2D_L1T<GridExpansionMode::FrontierParallel>;
using DBSCANGrid2D_L1UnionFind = DBSCANGrid2D_L1T<GridExpansionMode::UnionFind>;

} // namespace dbscan

