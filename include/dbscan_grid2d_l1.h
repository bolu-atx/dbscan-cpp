#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "perf_timer.h"

namespace dbscan {

struct DBSCANGrid2D_L1 {
  uint32_t eps;
  uint32_t min_samples;
  std::size_t num_threads;
  std::size_t chunk_size;

  DBSCANGrid2D_L1(uint32_t eps_value, uint32_t min_samples_value, std::size_t num_threads_value = 0,
                  std::size_t chunk_size_value = 0);

  [[nodiscard]] std::vector<int32_t> fit_predict(const uint32_t *x, const uint32_t *y, std::size_t count);

  PerfTiming perf_timing_;
};

} // namespace dbscan
