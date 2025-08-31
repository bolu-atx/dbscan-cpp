#pragma once

#include <vector>
#include <cstdint>
#include <optional>
#include <cmath>

namespace dbscan {

template<typename T = double>
struct Point {
    T x, y;
};

template<typename T = double>
struct ClusterResult {
    std::vector<int32_t> labels;  // -1 for noise, cluster id for core/border points
    int32_t num_clusters;
};

template<typename T = double>
class DBSCAN {
public:
    DBSCAN(T eps, int32_t min_pts);

    ClusterResult<T> cluster(const std::vector<Point<T> >& points) const;

private:
    T eps_;
    int32_t min_pts_;

    // Helper functions
    std::vector<int32_t> find_neighbors(const std::vector<Point<T> >& points, int32_t point_idx) const;
    T distance_squared(const Point<T>& a, const Point<T>& b) const;
    void expand_cluster(const std::vector<Point<T> >& points,
                       std::vector<int32_t>& labels,
                       int32_t point_idx,
                       int32_t cluster_id,
                       const std::vector<int32_t>& neighbors) const;
};

} // namespace dbscan