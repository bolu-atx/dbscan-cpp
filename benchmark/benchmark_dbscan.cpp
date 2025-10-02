#include "dbscan_grid2d_l1.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <nanobench.h>
#include <random>
#include <string>
#include <vector>

namespace {

struct Uint32Dataset {
  std::vector<uint32_t> x;
  std::vector<uint32_t> y;
};

Uint32Dataset generate_uint32_dataset(std::size_t cluster_count, std::size_t points_per_cluster,
                                      std::size_t noise_points, uint32_t area_width, uint32_t cluster_sigma,
                                      std::mt19937 &rng) {
  std::uniform_real_distribution<double> uniform_dist(0.0, static_cast<double>(area_width));
  std::normal_distribution<double> normal_dist(0.0, static_cast<double>(cluster_sigma));

  Uint32Dataset dataset;
  dataset.x.reserve(cluster_count * points_per_cluster + noise_points);
  dataset.y.reserve(cluster_count * points_per_cluster + noise_points);

  for (std::size_t c = 0; c < cluster_count; ++c) {
    const double center_x = uniform_dist(rng);
    const double center_y = uniform_dist(rng);

    for (std::size_t i = 0; i < points_per_cluster; ++i) {
      const double sample_x = center_x + normal_dist(rng);
      const double sample_y = center_y + normal_dist(rng);

      const uint32_t clamped_x =
          static_cast<uint32_t>(std::min(static_cast<double>(area_width - 1), std::max(0.0, std::round(sample_x))));
      const uint32_t clamped_y =
          static_cast<uint32_t>(std::min(static_cast<double>(area_width - 1), std::max(0.0, std::round(sample_y))));

      dataset.x.push_back(clamped_x);
      dataset.y.push_back(clamped_y);
    }
  }

  std::uniform_int_distribution<uint32_t> uniform_int(0, area_width - 1);
  for (std::size_t i = 0; i < noise_points; ++i) {
    dataset.x.push_back(uniform_int(rng));
    dataset.y.push_back(uniform_int(rng));
  }

  return dataset;
}

} // namespace

int main() {
  constexpr uint32_t area_width = 1'000'000;
  constexpr uint32_t cluster_sigma = 50; // Approximately 3 sigma ~ 150 px footprint
  constexpr uint32_t eps = 60;
  constexpr uint32_t min_samples = 16;

  std::mt19937 rng(1337u);
  ankerl::nanobench::Bench bench;
  bench.title("DBSCANGrid2D_L1");
  bench.relative(true);
  bench.warmup(2);
  bench.minEpochIterations(10);
  bench.unit("pt");

  struct Scenario {
    std::size_t clusters;
    std::size_t points_per_cluster;
  };

  const std::vector<Scenario> scenarios = {
      {64, 256},  // ~16K cluster points + 32K noise => ~48K total
      {128, 256}, // ~32K cluster points + 64K noise => ~96K total
      {256, 256}, // ~65K cluster points + 131K noise => ~196K total
      {512, 256}, // ~131K cluster points + 262K noise => ~393K total
      {640, 256}, // ~163K cluster points + 327K noise => ~490K total
  };

  std::cout << "Benchmarking DBSCANGrid2D_L1 with Manhattan distance" << std::endl;
  std::cout << "eps=" << eps << ", min_samples=" << min_samples << std::endl;
  std::cout << "Thread sweep: 0 (auto), 1, 2, 4, 8" << std::endl;

  for (const auto &scenario : scenarios) {
    const std::size_t cluster_points = scenario.clusters * scenario.points_per_cluster;
    const std::size_t noise_points = cluster_points * 2; // 2x noise compared to clustered points

    auto dataset = generate_uint32_dataset(scenario.clusters, scenario.points_per_cluster, noise_points, area_width,
                                           cluster_sigma, rng);

    const std::size_t total_points = dataset.x.size();
    std::cout << "\nScenario: " << scenario.clusters << " clusters, " << scenario.points_per_cluster
              << " points/cluster, total points=" << total_points << std::endl;

    bench.batch(static_cast<double>(total_points));
    bench.context("points", std::to_string(total_points));

    const std::vector<std::size_t> thread_counts = {0, 1, 2, 4, 8};
    for (std::size_t thread_count : thread_counts) {
      const std::string label = "grid-l1 " + std::to_string(total_points) + " pts threads=" +
                                (thread_count == 0 ? std::string("auto") : std::to_string(thread_count));
      bench.run(label, [&]() {
        dbscan::DBSCANGrid2DL1Params params{eps, min_samples};
        params.num_threads = thread_count;
        auto result =
            dbscan::dbscan_grid2d_l1(dataset.x.data(), 1, dataset.y.data(), 1, total_points, params);
        ankerl::nanobench::doNotOptimizeAway(result.labels);
      });
    }
  }

  return 0;
}
