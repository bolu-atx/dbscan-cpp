#include "dbscan_grid2d_l1.h"

#include <cstddef>
#include <stdexcept>

namespace dbscan {

DBSCANGrid2DL1Result dbscan_grid2d_l1_aos(const Grid2DPoint *points, std::size_t count,
                                          const DBSCANGrid2DL1Params &params, GridExpansionMode mode) {
  if (params.eps == 0)
    throw std::invalid_argument("eps must be greater than zero for dbscan_grid2d_l1_aos");
  if (params.min_samples == 0)
    throw std::invalid_argument("min_samples must be greater than zero for dbscan_grid2d_l1_aos");
  if (count != 0 && points == nullptr)
    throw std::invalid_argument("Input point array must be non-null when count > 0");

  static_assert(sizeof(Grid2DPoint) == sizeof(uint32_t) * 2, "Grid2DPoint is expected to be tightly packed");
  const std::size_t stride = sizeof(Grid2DPoint) / sizeof(uint32_t);

  const uint32_t *x = count == 0 ? nullptr : &points[0].x;
  const uint32_t *y = count == 0 ? nullptr : &points[0].y;
  return dbscan_grid2d_l1(x, stride, y, stride, count, params, mode);
}

} // namespace dbscan
