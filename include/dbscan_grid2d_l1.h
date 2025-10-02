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

struct DBSCANGrid2DL1Params {
  uint32_t eps;
  uint32_t min_samples;
  std::size_t num_threads = 0;
  std::size_t chunk_size = 0;
};

struct Grid2DPoint {
  uint32_t x;
  uint32_t y;
};

struct DBSCANGrid2DL1Result {
  std::vector<int32_t> labels;
  PerfTiming perf_timing;
};

[[nodiscard]] DBSCANGrid2DL1Result dbscan_grid2d_l1(const uint32_t *x, std::size_t x_stride, const uint32_t *y,
                                                    std::size_t y_stride, std::size_t count,
                                                    const DBSCANGrid2DL1Params &params,
                                                    GridExpansionMode mode = GridExpansionMode::Sequential);

[[nodiscard]] DBSCANGrid2DL1Result dbscan_grid2d_l1_aos(const Grid2DPoint *points, std::size_t count,
                                                        const DBSCANGrid2DL1Params &params,
                                                        GridExpansionMode mode = GridExpansionMode::Sequential);

} // namespace dbscan
