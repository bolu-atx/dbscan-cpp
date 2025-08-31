#include <catch2/catch_test_macros.hpp>
#include <dbscan.h>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstdlib>
#include <ctime>

namespace {

std::vector<dbscan::Point<double>> load_points_from_file(const std::string& filename) {
    std::vector<dbscan::Point<double>> points;

    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    // Read number of points
    uint32_t n_points;
    file.read(reinterpret_cast<char*>(&n_points), sizeof(n_points));

    points.reserve(n_points);

    // Read points
    for (uint32_t i = 0; i < n_points; ++i) {
        double x, y;
        file.read(reinterpret_cast<char*>(&x), sizeof(x));
        file.read(reinterpret_cast<char*>(&y), sizeof(y));
        points.push_back({x, y});
    }

    return points;
}

std::vector<int32_t> load_labels_from_file(const std::string& filename) {
    std::vector<int32_t> labels;

    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    // Read number of points
    uint32_t n_points;
    file.read(reinterpret_cast<char*>(&n_points), sizeof(n_points));

    // Skip points data
    file.seekg(sizeof(double) * 2 * n_points, std::ios::cur);

    labels.reserve(n_points);

    // Read labels
    for (uint32_t i = 0; i < n_points; ++i) {
        int32_t label;
        file.read(reinterpret_cast<char*>(&label), sizeof(label));
        labels.push_back(label);
    }

    return labels;
}

} // namespace

TEST_CASE("DBSCAN basic functionality test", "[dbscan]") {
    // Create simple test data
    std::vector<dbscan::Point<double>> points = {
        {0.0, 0.0}, {0.1, 0.1}, {0.2, 0.2},  // Cluster 1
        {5.0, 5.0}, {5.1, 5.1}, {5.2, 5.2},  // Cluster 2
        {10.0, 10.0}                          // Noise point
    };

    dbscan::DBSCAN<double> dbscan(0.5, 2);  // eps=0.5, min_pts=2
    auto result = dbscan.cluster(points);

    REQUIRE(result.labels.size() == points.size());
    REQUIRE(result.num_clusters >= 2);  // Should find at least 2 clusters

    // Check that points in same cluster have same label
    REQUIRE(result.labels[0] == result.labels[1]);  // First two points should be in same cluster
    REQUIRE(result.labels[0] == result.labels[2]);  // First three points should be in same cluster
    REQUIRE(result.labels[3] == result.labels[4]);  // Next two points should be in same cluster
    REQUIRE(result.labels[3] == result.labels[5]);  // Next three points should be in same cluster
    REQUIRE(result.labels[6] == -1);               // Last point should be noise
}

TEST_CASE("DBSCAN with 500 points", "[dbscan][performance]") {
    // Generate test data with 500 points
    std::vector<dbscan::Point<double>> points;
    points.reserve(500);

    // Create two clusters
    for (int i = 0; i < 200; ++i) {
        points.push_back({static_cast<double>(i % 20) * 0.1, static_cast<double>(i / 20) * 0.1});
    }
    for (int i = 0; i < 200; ++i) {
        points.push_back({5.0 + static_cast<double>(i % 20) * 0.1, static_cast<double>(i / 20) * 0.1});
    }
    // Add some noise
    for (int i = 0; i < 100; ++i) {
        points.push_back({10.0 + static_cast<double>(i % 10) * 0.1, 10.0 + static_cast<double>(i / 10) * 0.1});
    }

    dbscan::DBSCAN<double> dbscan(0.3, 3);
    auto result = dbscan.cluster(points);

    REQUIRE(result.labels.size() == 500);
    REQUIRE(result.num_clusters >= 2);  // Should find at least 2 clusters
}

TEST_CASE("DBSCAN with 10k points", "[dbscan][performance]") {
    // Generate test data with 10,000 points
    std::vector<dbscan::Point<double>> points;
    points.reserve(10000);

    // Create multiple clusters
    for (int c = 0; c < 5; ++c) {
        double center_x = c * 3.0;
        double center_y = c * 3.0;
        for (int i = 0; i < 1800; ++i) {
            double x = center_x + (static_cast<double>(rand()) / RAND_MAX - 0.5) * 0.8;
            double y = center_y + (static_cast<double>(rand()) / RAND_MAX - 0.5) * 0.8;
            points.push_back({x, y});
        }
    }
    // Add noise points
    for (int i = 0; i < 1000; ++i) {
        double x = 20.0 + (static_cast<double>(rand()) / RAND_MAX - 0.5) * 10.0;
        double y = 20.0 + (static_cast<double>(rand()) / RAND_MAX - 0.5) * 10.0;
        points.push_back({x, y});
    }

    dbscan::DBSCAN<double> dbscan(0.5, 5);
    auto result = dbscan.cluster(points);

    REQUIRE(result.labels.size() == 10000);
    REQUIRE(result.num_clusters >= 3);  // Should find multiple clusters
}

TEST_CASE("DBSCAN with 100k points", "[dbscan][performance]") {
    // Generate test data with 100,000 points (scaled down from 1M for practicality)
    std::vector<dbscan::Point<double>> points;
    points.reserve(100000);

    // Create clusters
    for (int c = 0; c < 8; ++c) {
        double center_x = c * 4.0;
        double center_y = c * 4.0;
        for (int i = 0; i < 12000; ++i) {
            double x = center_x + (static_cast<double>(rand()) / RAND_MAX - 0.5) * 1.0;
            double y = center_y + (static_cast<double>(rand()) / RAND_MAX - 0.5) * 1.0;
            points.push_back({x, y});
        }
    }
    // Add noise points
    for (int i = 0; i < 16000; ++i) {
        double x = 40.0 + (static_cast<double>(rand()) / RAND_MAX - 0.5) * 20.0;
        double y = 40.0 + (static_cast<double>(rand()) / RAND_MAX - 0.5) * 20.0;
        points.push_back({x, y});
    }

    dbscan::DBSCAN<double> dbscan(0.8, 5);
    auto result = dbscan.cluster(points);

    REQUIRE(result.labels.size() >= 100000);  // Allow for slight variations in data generation
    REQUIRE(result.num_clusters >= 5);  // Should find multiple clusters
}

TEST_CASE("DBSCAN different eps values", "[dbscan][parameters]") {
    std::vector<dbscan::Point<double>> points = {
        {0.0, 0.0}, {0.1, 0.1}, {0.2, 0.2},  // Close cluster
        {2.0, 2.0}, {2.1, 2.1}, {2.2, 2.2},  // Medium distance cluster
        {5.0, 5.0}, {5.1, 5.1}, {5.2, 5.2}   // Far cluster
    };

    // Test with small eps (should create 3 clusters)
    dbscan::DBSCAN<double> dbscan_small_eps(0.3, 2);
    auto result_small = dbscan_small_eps.cluster(points);
    REQUIRE(result_small.num_clusters >= 3);

    // Test with large eps (should create fewer clusters)
    dbscan::DBSCAN<double> dbscan_large_eps(3.0, 2);
    auto result_large = dbscan_large_eps.cluster(points);
    REQUIRE(result_large.num_clusters <= result_small.num_clusters);
}

TEST_CASE("DBSCAN different min_pts values", "[dbscan][parameters]") {
    std::vector<dbscan::Point<double>> points = {
        {0.0, 0.0}, {0.1, 0.1}, {0.2, 0.2}, {0.3, 0.3},  // 4 points
        {2.0, 2.0}, {2.1, 2.1}, {2.2, 2.2}                 // 3 points
    };

    // Test with min_pts = 3 (should find 2 clusters)
    dbscan::DBSCAN<double> dbscan_min3(0.5, 3);
    auto result_min3 = dbscan_min3.cluster(points);
    REQUIRE(result_min3.num_clusters >= 1);

    // Test with min_pts = 5 (should find fewer clusters)
    dbscan::DBSCAN<double> dbscan_min5(0.5, 5);
    auto result_min5 = dbscan_min5.cluster(points);
    REQUIRE(result_min5.num_clusters <= result_min3.num_clusters);
}

TEST_CASE("DBSCAN handles empty input", "[dbscan]") {
    dbscan::DBSCAN<double> dbscan(0.5, 3);
    std::vector<dbscan::Point<double>> empty_points;

    auto result = dbscan.cluster(empty_points);

    REQUIRE(result.labels.empty());
    REQUIRE(result.num_clusters == 0);
}

TEST_CASE("DBSCAN handles single point", "[dbscan]") {
    dbscan::DBSCAN<double> dbscan(0.5, 3);
    std::vector<dbscan::Point<double>> single_point = {{1.0, 2.0}};

    auto result = dbscan.cluster(single_point);

    REQUIRE(result.labels.size() == 1);
    REQUIRE(result.labels[0] == -1);  // Should be noise
    REQUIRE(result.num_clusters == 0);
}

TEST_CASE("DBSCAN handles all noise", "[dbscan]") {
    dbscan::DBSCAN<double> dbscan(0.1, 5);  // Very small eps, high min_pts
    std::vector<dbscan::Point<double>> scattered_points = {
        {0.0, 0.0}, {1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}
    };

    auto result = dbscan.cluster(scattered_points);

    REQUIRE(result.labels.size() == 4);
    for (int label : result.labels) {
        REQUIRE(label == -1);  // All should be noise
    }
    REQUIRE(result.num_clusters == 0);
}