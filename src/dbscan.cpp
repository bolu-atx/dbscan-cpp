#include "dbscan.h"
#include <queue>
#include <vector>
#include <cmath>

namespace dbscan {

template<typename T>
DBSCAN<T>::DBSCAN(T eps, int32_t min_pts)
    : eps_(eps), min_pts_(min_pts) {}

template<typename T>
ClusterResult<T> DBSCAN<T>::cluster(const std::vector<Point<T> >& points) const {
    if (points.empty()) {
        return {{}, 0};
    }

    std::vector<int32_t> labels(points.size(), -1);  // -1 means unvisited
    int32_t cluster_id = 0;

    for (int32_t i = 0; i < static_cast<int32_t>(points.size()); ++i) {
        if (labels[i] != -1) continue;  // Already processed

        auto neighbors = find_neighbors(points, i);

        if (static_cast<int32_t>(neighbors.size()) < min_pts_) {
            labels[i] = -2;  // Mark as noise for now
        } else {
            expand_cluster(points, labels, i, cluster_id, neighbors);
            ++cluster_id;
        }
    }

    // Convert noise markers back to -1
    for (auto& label : labels) {
        if (label == -2) label = -1;
    }

    return {labels, cluster_id};
}

template<typename T>
std::vector<int32_t> DBSCAN<T>::find_neighbors(const std::vector<Point<T> >& points, int32_t point_idx) const {
    std::vector<int32_t> neighbors;
    const Point<T>& target = points[point_idx];
    T eps_squared = eps_ * eps_;

    for (size_t i = 0; i < points.size(); ++i) {
        if (i == static_cast<size_t>(point_idx)) continue;

        T dx = points[i].x - target.x;
        T dy = points[i].y - target.y;
        T dist_squared = dx * dx + dy * dy;

        if (dist_squared <= eps_squared) {
            neighbors.push_back(static_cast<int32_t>(i));
        }
    }

    return neighbors;
}

template<typename T>
T DBSCAN<T>::distance_squared(const Point<T>& a, const Point<T>& b) const {
    T dx = a.x - b.x;
    T dy = a.y - b.y;
    return dx * dx + dy * dy;
}

template<typename T>
void DBSCAN<T>::expand_cluster(const std::vector<Point<T> >& points,
                              std::vector<int32_t>& labels,
                              int32_t point_idx,
                              int32_t cluster_id,
                              const std::vector<int32_t>& neighbors) const {
    labels[point_idx] = cluster_id;

    std::queue<int32_t> seeds;
    for (int32_t neighbor : neighbors) {
        seeds.push(neighbor);
    }

    while (!seeds.empty()) {
        int32_t current_idx = seeds.front();
        seeds.pop();

        if (labels[current_idx] == -2) {
            // Previously marked as noise, now it's a border point
            labels[current_idx] = cluster_id;
        }

        if (labels[current_idx] != -1) continue;  // Already processed

        labels[current_idx] = cluster_id;

        auto current_neighbors = find_neighbors(points, current_idx);
        if (static_cast<int32_t>(current_neighbors.size()) >= min_pts_) {
            // Current point is a core point, add its neighbors to seeds
            for (int32_t neighbor : current_neighbors) {
                if (labels[neighbor] == -1 || labels[neighbor] == -2) {
                    seeds.push(neighbor);
                }
            }
        }
    }
}

// Explicit template instantiations
template class DBSCAN<double>;
template class DBSCAN<float>;

} // namespace dbscan