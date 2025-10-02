#include "dbscan_grid2d_l1.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

namespace {

std::filesystem::path fixture_root() {
  const std::filesystem::path candidates[] = {std::filesystem::path{"tests"} / "data",
                                              std::filesystem::path{".."} / "tests" / "data",
                                              std::filesystem::path{".."} / ".." / "tests" / "data"};
  for (const auto &candidate : candidates) {
    if (std::filesystem::exists(candidate))
      return candidate;
  }
  return candidates[0];
}

// The binary fixtures store Y first then X to mirror the grid compaction used in production; this loader
// preserves that ordering so the test exercises identical memory access patterns as the runtime path.
void load_fixture(const std::filesystem::path &data_path, const std::filesystem::path &truth_path,
                  std::vector<uint32_t> &x_out, std::vector<uint32_t> &y_out, std::vector<int32_t> &truth_out) {
  std::ifstream data_stream(data_path, std::ios::binary);
  REQUIRE(data_stream.good());

  data_stream.seekg(0, std::ios::end);
  const auto data_size = data_stream.tellg();
  data_stream.seekg(0, std::ios::beg);

  REQUIRE(data_size % (sizeof(uint32_t) * 2) == 0);
  const std::size_t point_count = static_cast<std::size_t>(data_size) / (sizeof(uint32_t) * 2);

  x_out.assign(point_count, 0U);
  y_out.assign(point_count, 0U);

  for (std::size_t i = 0; i < point_count; ++i) {
    uint32_t y = 0;
    uint32_t x = 0;
    data_stream.read(reinterpret_cast<char *>(&y), sizeof(uint32_t));
    data_stream.read(reinterpret_cast<char *>(&x), sizeof(uint32_t));
    REQUIRE(data_stream.good());
    y_out[i] = y;
    x_out[i] = x;
  }

  std::ifstream truth_stream(truth_path, std::ios::binary);
  REQUIRE(truth_stream.good());

  truth_stream.seekg(0, std::ios::end);
  const auto truth_size = truth_stream.tellg();
  truth_stream.seekg(0, std::ios::beg);

  REQUIRE(truth_size % sizeof(int32_t) == 0);
  const std::size_t truth_count = static_cast<std::size_t>(truth_size) / sizeof(int32_t);
  REQUIRE(truth_count == point_count);

  truth_out.assign(truth_count, 0);
  truth_stream.read(reinterpret_cast<char *>(truth_out.data()),
                    static_cast<std::streamsize>(truth_count * sizeof(int32_t)));
  REQUIRE(truth_stream.good());
}

std::vector<dbscan::Grid2DPoint> make_aos(const std::vector<uint32_t> &x, const std::vector<uint32_t> &y) {
  std::vector<dbscan::Grid2DPoint> points(x.size());
  for (std::size_t i = 0; i < points.size(); ++i)
    points[i] = dbscan::Grid2DPoint{x[i], y[i]};
  return points;
}

} // namespace

TEST_CASE("DBSCANGrid2D_L1 clusters dense neighbors", "[dbscan][grid_l1]") {
  const std::vector<uint32_t> x = {0, 1, 2, 100};
  const std::vector<uint32_t> y = {0, 0, 1, 200};

  const dbscan::DBSCANGrid2DL1Params params{4, 3};
  const auto soa_result = dbscan::dbscan_grid2d_l1(x.data(), 1, y.data(), 1, x.size(), params);
  const auto &labels = soa_result.labels;

  REQUIRE(labels.size() == x.size());
  REQUIRE(labels[0] == labels[1]);
  REQUIRE(labels[1] == labels[2]);
  REQUIRE(labels[0] != -1);
  REQUIRE(labels[3] == -1);

  const auto aos_points = make_aos(x, y);
  const auto aos_result = dbscan::dbscan_grid2d_l1_aos(aos_points.data(), aos_points.size(), params);
  REQUIRE(aos_result.labels == labels);
}

TEST_CASE("DBSCANGrid2D_L1 respects min_samples threshold", "[dbscan][grid_l1]") {
  const std::vector<uint32_t> coords = {0, 2, 4};
  const dbscan::DBSCANGrid2DL1Params params{3, 4};
  const auto result = dbscan::dbscan_grid2d_l1(coords.data(), 1, coords.data(), 1, coords.size(), params);
  REQUIRE(result.labels.size() == coords.size());
  for (int32_t label : result.labels) {
    REQUIRE(label == -1);
  }
}

TEST_CASE("DBSCANGrid2D_L1 matches fixture truth", "[dbscan][grid_l1]") {
  const std::filesystem::path root = fixture_root();
  const auto data_path = root / "dbscan_static_data.bin";
  const auto truth_path = root / "dbscan_static_truth.bin";

  std::vector<uint32_t> x;
  std::vector<uint32_t> y;
  std::vector<int32_t> truth;
  load_fixture(data_path, truth_path, x, y, truth);

  const dbscan::DBSCANGrid2DL1Params params{10, 3};
  const auto sequential = dbscan::dbscan_grid2d_l1(x.data(), 1, y.data(), 1, x.size(), params);
  REQUIRE(sequential.labels == truth);

  const auto frontier =
      dbscan::dbscan_grid2d_l1(x.data(), 1, y.data(), 1, x.size(), params, dbscan::GridExpansionMode::FrontierParallel);
  REQUIRE(frontier.labels == truth);

  const auto union_find =
      dbscan::dbscan_grid2d_l1(x.data(), 1, y.data(), 1, x.size(), params, dbscan::GridExpansionMode::UnionFind);
  REQUIRE(union_find.labels == truth);
}

TEST_CASE("DBSCANGrid2D_L1 parallel variants align with sequential", "[dbscan][grid_l1][parallel]") {
  const std::vector<uint32_t> x = {0, 1, 2, 5, 40};
  const std::vector<uint32_t> y = {0, 0, 1, 4, 80};

  dbscan::DBSCANGrid2DL1Params params{6, 3};
  params.num_threads = 4;

  const auto sequential = dbscan::dbscan_grid2d_l1(x.data(), 1, y.data(), 1, x.size(), params);

  const auto frontier =
      dbscan::dbscan_grid2d_l1(x.data(), 1, y.data(), 1, x.size(), params, dbscan::GridExpansionMode::FrontierParallel);
  REQUIRE(frontier.labels == sequential.labels);

  const auto union_find =
      dbscan::dbscan_grid2d_l1(x.data(), 1, y.data(), 1, x.size(), params, dbscan::GridExpansionMode::UnionFind);
  REQUIRE(union_find.labels == sequential.labels);
}
