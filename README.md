# DBSCAN C++ Implementation

A high-performance, templated C++20 implementation of the DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm.

## Features

- **C++20 compliant** with modern syntax and features
- **Templated implementation** supporting different data types (double, float, etc.)
- **2D data clustering** optimized for spatial data
- **No external dependencies** except Catch2 for testing
- **CMake build system** for easy compilation
- **Comprehensive test suite** with performance benchmarks
- **Binary file I/O** for efficient data handling

## Requirements

- C++20 compatible compiler (GCC 10+, Clang 10+, MSVC 2019+)
- CMake 3.20 or higher
- Git (for cloning and development)

## Building

```bash
# Clone the repository
git clone https://github.com/yourusername/dbscan-cpp.git
cd dbscan-cpp

# Create build directory
mkdir build && cd build

# Configure and build
cmake ..
make -j$(nproc)

# Run tests
./dbscan_tests
```

## Usage

### Basic Usage

```cpp
#include "dbscan.h"
#include <vector>

// Create 2D points
std::vector<dbscan::Point<double>> points = {
    {0.0, 0.0}, {0.1, 0.1}, {0.2, 0.2},  // Cluster 1
    {5.0, 5.0}, {5.1, 5.1}, {5.2, 5.2},  // Cluster 2
    {10.0, 10.0}                          // Noise point
};

// Run DBSCAN clustering
dbscan::DBSCAN<double> dbscan(0.5, 2);  // eps=0.5, min_pts=2
auto result = dbscan.cluster(points);

// Access results
std::cout << "Number of clusters: " << result.num_clusters << std::endl;
for (size_t i = 0; i < result.labels.size(); ++i) {
    std::cout << "Point " << i << ": Cluster " << result.labels[i] << std::endl;
}
```

### Template Specialization

```cpp
// Use float precision
dbscan::DBSCAN<float> dbscan_float(0.5f, 2);
std::vector<dbscan::Point<float>> float_points = {{0.0f, 0.0f}, {0.1f, 0.1f}};
auto result_float = dbscan_float.cluster(float_points);
```

## API Reference

### Classes

#### `dbscan::Point<T>`
Represents a 2D point with coordinates of type `T`.

```cpp
template<typename T = double>
struct Point {
    T x, y;
};
```

#### `dbscan::DBSCAN<T>`
Main DBSCAN clustering class.

```cpp
template<typename T = double>
class DBSCAN {
public:
    DBSCAN(T eps, int32_t min_pts);
    ClusterResult<T> cluster(const std::vector<Point<T>>& points) const;
};
```

#### `dbscan::ClusterResult<T>`
Contains clustering results.

```cpp
template<typename T = double>
struct ClusterResult {
    std::vector<int32_t> labels;  // -1 for noise, cluster id for core/border points
    int32_t num_clusters;
};
```

## Parameters

- **eps**: Maximum distance between two points to be considered neighbors
- **min_pts**: Minimum number of points required to form a dense region (core point)

## Performance

The implementation has been tested with datasets up to 100,000 points and shows good performance characteristics:

- **500 points**: ~instantaneous
- **10,000 points**: < 1 second
- **100,000 points**: < 10 seconds (depending on parameters)

## Testing

The project includes a comprehensive test suite covering:

- Basic functionality verification
- Performance testing with different data sizes
- Parameter sensitivity testing
- Edge cases (empty input, single point, all noise)

```bash
# Run all tests
./dbscan_tests

# Run specific test
./dbscan_tests -t "DBSCAN with 10k points"
```

## Data Format

The project includes utilities for reading/writing binary data files:

```cpp
// Generate test data
python3 generate_test_data.py --multiple

// Load binary data
auto points = load_points_from_file("test_data.bin");
```

Binary format:
- 4 bytes: number of points (uint32_t)
- For each point: 8 bytes x coordinate + 8 bytes y coordinate (double)
- For each point: 4 bytes cluster label (int32_t)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- DBSCAN algorithm by Martin Ester, Hans-Peter Kriegel, Jörg Sander, and Xiaowei Xu
- Catch2 testing framework
- CMake build system