#include "dbscan.h"
#include <iostream>
#include <vector>

int main() {
    // Create sample 2D data points
    std::vector<dbscan::Point<double>> points = {
        {0.0, 0.0}, {0.1, 0.1}, {0.2, 0.2},  // Cluster 1
        {5.0, 5.0}, {5.1, 5.1}, {5.2, 5.2},  // Cluster 2
        {10.0, 10.0}                          // Noise point
    };

    // Run DBSCAN clustering
    dbscan::DBSCAN<double> dbscan(0.5, 2);  // eps=0.5, min_pts=2
    auto result = dbscan.cluster(points);

    // Print results
    std::cout << "DBSCAN Clustering Results:" << std::endl;
    std::cout << "Number of clusters found: " << result.num_clusters << std::endl;
    std::cout << "Point classifications:" << std::endl;

    for (size_t i = 0; i < points.size(); ++i) {
        std::cout << "Point (" << points[i].x << ", " << points[i].y << "): ";
        if (result.labels[i] == -1) {
            std::cout << "NOISE" << std::endl;
        } else {
            std::cout << "Cluster " << result.labels[i] << std::endl;
        }
    }

    return 0;
}