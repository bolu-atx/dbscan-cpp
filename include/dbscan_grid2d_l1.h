#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace dbscan {

struct DBSCANGrid2D_L1 {
  uint32_t eps;
  uint32_t min_samples;

  DBSCANGrid2D_L1(uint32_t eps_value, uint32_t min_samples_value);

  [[nodiscard]] std::vector<int32_t> fit_predict(const uint32_t *x, const uint32_t *y, std::size_t count) const;
};

} // namespace dbscan
