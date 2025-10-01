#include "dbscan_grid2d_l1.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

namespace {

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

} // namespace

TEST_CASE("DBSCANGrid2D_L1 clusters dense neighbors", "[dbscan][grid_l1]") {
  // Points are arranged so the Manhattan frontier just connects the first three but leaves the outlier isolated,
  // validating that the L1 grid expansion covers diagonals without absorbing distant noise.
  const std::vector<uint32_t> x = {0, 1, 2, 100};
  const std::vector<uint32_t> y = {0, 0, 1, 200};

  dbscan::DBSCANGrid2D_L1 algo(4, 3);
  auto labels = algo.fit_predict(x.data(), y.data(), x.size());

  REQUIRE(labels.size() == x.size());
  REQUIRE(labels[0] == labels[1]);
  REQUIRE(labels[1] == labels[2]);
  REQUIRE(labels[0] != -1);
  REQUIRE(labels[3] == -1);
}

TEST_CASE("DBSCANGrid2D_L1 respects min_samples threshold", "[dbscan][grid_l1]") {
  // Every point is deliberately spaced just beyond eps so we confirm the min_samples guard suppresses tiny clusters.
  const std::vector<uint32_t> coords = {0, 2, 4};
  dbscan::DBSCANGrid2D_L1 algo(3, 4);
  auto labels = algo.fit_predict(coords.data(), coords.data(), coords.size());

  REQUIRE(labels.size() == coords.size());
  for (int32_t label : labels) {
    REQUIRE(label == -1);
  }
}

TEST_CASE("DBSCANGrid2D_L1 matches fixture truth", "[dbscan][grid_l1]") {
  // Fixture run mirrors the end-to-end validator to ensure the optimized grid path stays aligned with reference data.
  const std::filesystem::path root = std::filesystem::path{"tests"} / "data";
  const auto data_path = root / "dbscan_static_data.bin";
  const auto truth_path = root / "dbscan_static_truth.bin";

  std::vector<uint32_t> x;
  std::vector<uint32_t> y;
  std::vector<int32_t> truth;
  load_fixture(data_path, truth_path, x, y, truth);

  dbscan::DBSCANGrid2D_L1 algo(10, 3);
  auto labels = algo.fit_predict(x.data(), y.data(), x.size());

  REQUIRE(labels.size() == truth.size());
  REQUIRE(labels == truth);
}
