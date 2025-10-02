#include "dbscan_grid2d_l1.h"

#include <stdexcept>

#include "dbscan_grid2d_l1_impl.hpp"

namespace dbscan {

DBSCANGrid2DL1Result dbscan_grid2d_l1(const uint32_t *x, std::size_t x_stride, const uint32_t *y, std::size_t y_stride,
                                      std::size_t count, const DBSCANGrid2DL1Params &params, GridExpansionMode mode) {
  if (params.eps == 0)
    throw std::invalid_argument("eps must be greater than zero for dbscan_grid2d_l1");
  if (params.min_samples == 0)
    throw std::invalid_argument("min_samples must be greater than zero for dbscan_grid2d_l1");
  if (count != 0 && (x == nullptr || y == nullptr))
    throw std::invalid_argument("Input coordinate arrays must be non-null when count > 0");
  if (count != 0 && (x_stride == 0 || y_stride == 0))
    throw std::invalid_argument("Stride must be positive when count > 0");

  return detail::dbscan_grid2d_l1_impl(x, x_stride, y, y_stride, count, params, mode);
}

} // namespace dbscan
