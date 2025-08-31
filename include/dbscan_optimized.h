#pragma once

#include "dbscan.h"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <atomic>
#include <execution>
#include <cmath>

namespace dbscan {

template<typename T = double>
class UnionFind {
private:
    std::vector<int32_t> parent;
    std::vector<int32_t> rank;
    std::mutex mutex;

public:
    UnionFind(size_t size) : parent(size), rank(size, 0) {
        for (size_t i = 0; i < size; ++i) {
            parent[i] = static_cast<int32_t>(i);
        }
    }

    int32_t find(int32_t x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);
        }
        return parent[x];
    }

    void union_sets(int32_t x, int32_t y) {
        std::lock_guard<std::mutex> lock(mutex);
        int32_t root_x = find(x);
        int32_t root_y = find(y);

        if (root_x != root_y) {
            if (rank[root_x] < rank[root_y]) {
                parent[root_x] = root_y;
            } else if (rank[root_x] > rank[root_y]) {
                parent[root_y] = root_x;
            } else {
                parent[root_y] = root_x;
                rank[root_x]++;
            }
        }
    }

    std::vector<int32_t> get_labels() const {
        return parent;
    }
};

template<typename T = double>
struct GridCell {
    std::vector<size_t> points;
};

template<typename T = double>
class SpatialGrid {
private:
    T cell_size;
    size_t grid_width, grid_height;
    T min_x, min_y, max_x, max_y;
    std::vector<std::vector<GridCell<T> > > grid;

public:
    SpatialGrid(T eps, const std::vector<Point<T>>& points) : cell_size(eps) {
        if (points.empty()) return;

        // Find bounds
        min_x = max_x = points[0].x;
        min_y = max_y = points[0].y;

        for (const auto& point : points) {
            min_x = std::min(min_x, point.x);
            max_x = std::max(max_x, point.x);
            min_y = std::min(min_y, point.y);
            max_y = std::max(max_y, point.y);
        }

        // Add padding
        T padding = eps;
        min_x -= padding;
        min_y -= padding;
        max_x += padding;
        max_y += padding;

        // Calculate grid dimensions
        grid_width = static_cast<size_t>((max_x - min_x) / cell_size) + 1;
        grid_height = static_cast<size_t>((max_y - min_y) / cell_size) + 1;

        // Initialize grid
        grid.resize(grid_height, std::vector<GridCell<T> >(grid_width));

        // Insert points into grid
        for (size_t i = 0; i < points.size(); ++i) {
            size_t cell_x = static_cast<size_t>((points[i].x - min_x) / cell_size);
            size_t cell_y = static_cast<size_t>((points[i].y - min_y) / cell_size);

            if (cell_x < grid_width && cell_y < grid_height) {
                grid[cell_y][cell_x].points.push_back(i);
            }
        }
    }

    std::vector<std::pair<size_t, size_t>> get_neighbor_cells(size_t cell_x, size_t cell_y) const {
        std::vector<std::pair<size_t, size_t>> neighbors;

        // Check 3x3 neighborhood (including center cell)
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = static_cast<int>(cell_x) + dx;
                int ny = static_cast<int>(cell_y) + dy;

                if (nx >= 0 && nx < static_cast<int>(grid_width) &&
                    ny >= 0 && ny < static_cast<int>(grid_height)) {
                    neighbors.push_back(std::pair<size_t, size_t>(nx, ny));
                }
            }
        }

        return neighbors;
    }

    std::vector<size_t> get_points_in_cell(size_t cell_x, size_t cell_y) const {
        if (cell_y < grid_height && cell_x < grid_width) {
            return grid[cell_y][cell_x].points;
        }
        return std::vector<size_t>();
    }

    std::pair<size_t, size_t> get_cell_coords(const Point<T>& point) const {
        size_t cell_x = static_cast<size_t>((point.x - min_x) / cell_size);
        size_t cell_y = static_cast<size_t>((point.y - min_y) / cell_size);
        return std::pair<size_t, size_t>(cell_x, cell_y);
    }
};

template<typename T = double>
class DBSCANOptimized {
private:
    T eps_;
    int32_t min_pts_;
    SpatialGrid<T> grid_;
    std::vector<Point<T> > points_;
    size_t grid_width;

public:
    DBSCANOptimized(T eps, int32_t min_pts, const std::vector<Point<T> >& points)
        : eps_(eps), min_pts_(min_pts), grid_(eps, points), points_(points) {}

    ClusterResult<T> cluster();

private:
    std::vector<bool> find_core_points() const;
    std::vector<size_t> get_neighbors(size_t point_idx) const;
    T distance_squared(const Point<T>& a, const Point<T>& b) const;
    void process_core_core_connections(const std::vector<bool>& is_core,
                                     UnionFind<T>& uf) const;
    std::vector<int32_t> assign_border_points(const std::vector<bool>& is_core,
                                            const UnionFind<T>& uf) const;
    int32_t count_clusters(const UnionFind<T>& uf) const;
};

} // namespace dbscan