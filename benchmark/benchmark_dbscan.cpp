#include "dbscan.h"
#include "dbscan_optimized.h"
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <memory>
#include <nanobench.h>
#include <string>
#include <vector>

// Generate clustered 2D data for benchmarking
std::vector<dbscan::Point<double>> generate_benchmark_data(size_t n_points, int n_clusters = 8) {
  std::vector<dbscan::Point<double>> points;
  points.reserve(n_points);

  // Create clusters
  for (int c = 0; c < n_clusters; ++c) {
    double center_x = c * 5.0;
    double center_y = c * 5.0;
    size_t points_per_cluster = n_points / n_clusters;

    for (size_t i = 0; i < points_per_cluster; ++i) {
      double x = center_x + (static_cast<double>(rand()) / RAND_MAX - 0.5) * 2.0;
      double y = center_y + (static_cast<double>(rand()) / RAND_MAX - 0.5) * 2.0;
      points.push_back({x, y});
    }
  }

  // Add some noise points
  size_t noise_points = n_points / 10;
  for (size_t i = 0; i < noise_points; ++i) {
    double x = 50.0 + (static_cast<double>(rand()) / RAND_MAX - 0.5) * 20.0;
    double y = 50.0 + (static_cast<double>(rand()) / RAND_MAX - 0.5) * 20.0;
    points.push_back({x, y});
  }

  return points;
}

int main() {
  // Seed random number generator
  srand(static_cast<unsigned int>(time(nullptr)));

  ankerl::nanobench::Bench bench;

  // Benchmark different data sizes
  std::vector<size_t> data_sizes = {1000, 10000, 50000, 100000};

  for (size_t n_points : data_sizes) {
    std::cout << "\n=== Benchmarking with " << n_points << " points ===" << std::endl;

    // Generate test data
    auto points = generate_benchmark_data(n_points);

    // Benchmark original DBSCAN
    bench.title("Original DBSCAN").run("Original DBSCAN " + std::to_string(n_points) + " points", [&]() {
      dbscan::DBSCAN<double> dbscan(0.8, 5);
      auto result = dbscan.cluster(points);
      ankerl::nanobench::doNotOptimizeAway(result);
    });

    // Benchmark optimized DBSCAN
    bench.title("Optimized DBSCAN").run("Optimized DBSCAN " + std::to_string(n_points) + " points", [&]() {
      dbscan::DBSCANOptimized<double> dbscan(0.8, 5);
      auto result = dbscan.cluster(points);
      ankerl::nanobench::doNotOptimizeAway(result);
    });

    // Memory usage comparison
    {
      dbscan::DBSCAN<double> original_dbscan(0.8, 5);
      auto original_result = original_dbscan.cluster(points);

      dbscan::DBSCANOptimized<double> optimized_dbscan(0.8, 5);
      auto optimized_result = optimized_dbscan.cluster(points);

      std::cout << "Original DBSCAN found " << original_result.num_clusters << " clusters" << std::endl;
      std::cout << "Optimized DBSCAN found " << optimized_result.num_clusters << " clusters" << std::endl;
    }
  }

  // Performance comparison with different parameters
  std::cout << "\n=== Parameter Sensitivity Benchmark ===" << std::endl;

  auto test_points = generate_benchmark_data(10000);

  // Different eps values
  std::vector<double> eps_values = {0.3, 0.5, 0.8, 1.2};

  for (double eps : eps_values) {
    bench.title("EPS Parameter").run("Optimized DBSCAN eps=" + std::to_string(eps), [&]() {
      dbscan::DBSCANOptimized<double> dbscan(eps, 5);
      auto result = dbscan.cluster(test_points);
      ankerl::nanobench::doNotOptimizeAway(result);
    });
  }

  // Different min_pts values
  std::vector<int> min_pts_values = {3, 5, 10, 15};

  for (int min_pts : min_pts_values) {
    bench.title("MinPts Parameter").run("Optimized DBSCAN min_pts=" + std::to_string(min_pts), [&]() {
      dbscan::DBSCANOptimized<double> dbscan(0.8, min_pts);
      auto result = dbscan.cluster(test_points);
      ankerl::nanobench::doNotOptimizeAway(result);
    });
  }

  // Detailed performance analysis
  std::cout << "\n=== Detailed Performance Analysis ===" << std::endl;

  auto large_dataset = generate_benchmark_data(50000);

  // Time both implementations on larger dataset
  {
    std::cout << "Running performance comparison on 50k points..." << std::endl;

    // Original DBSCAN timing
    auto start_time = std::chrono::high_resolution_clock::now();
    dbscan::DBSCAN<double> original_dbscan(0.8, 5);
    auto original_result = original_dbscan.cluster(large_dataset);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto original_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Optimized DBSCAN timing
    start_time = std::chrono::high_resolution_clock::now();
    dbscan::DBSCANOptimized<double> optimized_dbscan(0.8, 5);
    auto optimized_result = optimized_dbscan.cluster(large_dataset);
    end_time = std::chrono::high_resolution_clock::now();
    auto optimized_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "Original DBSCAN: " << original_duration.count() << "ms, " << original_result.num_clusters
              << " clusters" << std::endl;
    std::cout << "Optimized DBSCAN: " << optimized_duration.count() << "ms, " << optimized_result.num_clusters
              << " clusters" << std::endl;

    if (original_duration.count() > 0) {
      double speedup = static_cast<double>(original_duration.count()) / optimized_duration.count();
      std::cout << "Speedup: " << speedup << "x" << std::endl;
    }
  }

  return 0;
}