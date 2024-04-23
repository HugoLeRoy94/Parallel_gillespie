//g++ -shared -o clustering.so -fPIC -O3 clustering.cpp
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <set>

// Euclidean distance calculation
float euclidean_distance(const std::vector<float>& a, const std::vector<float>& b) {
    float sum = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

extern "C" {

    // Define a struct to return the average cluster size and distance between clusters
    struct ClusterStats {
        float average_cluster_size;
        float average_distance_between_clusters;
    };

    // Define the function that calculates these statistics
    ClusterStats hierarchical_clustering_with_stats(float* points, size_t num_points, float max_distance) {
        if (points == nullptr || num_points == 0) {
            ClusterStats empty_stats = {0.0, 0.0};
            return empty_stats;
        }

        // Convert raw pointer to vector of vectors
        std::vector<std::vector<float>> point_vector(num_points, std::vector<float>(3));
        for (size_t i = 0; i < num_points; i++) {
            point_vector[i][0] = points[i * 3 + 0];
            point_vector[i][1] = points[i * 3 + 1];
            point_vector[i][2] = points[i * 3 + 2];
        }

    // Initial clustering
        std::vector<std::vector<std::vector<float>>> clusters;
        for (const auto& point : point_vector) {
            bool added_to_cluster = false;
            for (auto& cluster : clusters) {
                for (const auto& cpoint : cluster) {
                    if (euclidean_distance(point, cpoint) <= max_distance) {
                        cluster.push_back(point);
                        added_to_cluster = true;
                        break; // Breaks only the inner loop
                    }
                }
                //if (added_to_cluster) {
                //    continue; // Breaks only the inner loop
                //}
            }
            if (!added_to_cluster) {
                clusters.push_back({ point });
            }
        }
        // Merge overlapping clusters
        std::vector<std::set<std::vector<float>>> clusters_as_sets;
        for (const auto& cluster : clusters) {
            std::set<std::vector<float>> cluster_set(cluster.begin(), cluster.end());
            clusters_as_sets.push_back(cluster_set);
        }
        std::vector<std::vector<std::vector<float>>> merged_clusters;
        while (!clusters_as_sets.empty()) {
            auto first = clusters_as_sets.front();
            clusters_as_sets.erase(clusters_as_sets.begin());

            bool merged = true;
            while (merged) {
                merged = false;
                auto it = clusters_as_sets.begin();
                while (it != clusters_as_sets.end()) {
                    bool overlaps = false;
                    for (const auto& fpoint : first) {
                        if (it->find(fpoint) != it->end()) {
                            overlaps = true;
                            break;
                        }
                    }

                    if (overlaps) {
                        first.insert(it->begin(), it->end());
                        it = clusters_as_sets.erase(it);
                        merged = true;
                    } else {
                        ++it;
                    }
                }
            }

            merged_clusters.push_back(std::vector<std::vector<float>>(first.begin(), first.end()));
        }

// Calculate average cluster size
        std::vector<float> cluster_sizes;
        for (const auto& cluster : merged_clusters) {
            cluster_sizes.push_back(cluster.size());
        }
        float average_cluster_size = std::accumulate(
            cluster_sizes.begin(), cluster_sizes.end(), 0.0
        ) / cluster_sizes.size();

        // Calculate centroids and distances between them
        std::vector<std::vector<float>> centroids;
        for (const auto& cluster : merged_clusters) {
            std::vector<float> centroid(3, 0.0);
            for (const auto& point : cluster) {
                for (size_t j = 0; j < 3; j++) {
                    centroid[j] += point[j];
                }
            }
            for (size_t j = 0; j < 3; j++) {
                centroid[j] /= cluster.size();
            }
            centroids.push_back(centroid);
        }

        float average_distance_between_clusters = 0.0;
        if (centroids.size() > 1) {
            std::vector<float> distances;
            for (size_t i = 0; i < centroids.size(); i++) {
                for (size_t j = i + 1; j < centroids.size(); j++) {
                    distances.push_back(euclidean_distance(centroids[i], centroids[j]));
                }
            }
            average_distance_between_clusters = std::accumulate(
                distances.begin(), distances.end(), 0.0
            ) / distances.size();
        }

        // Return the clustering result with the calculated statistics
        ClusterStats result;
        result.average_cluster_size = average_cluster_size;
        result.average_distance_between_clusters = average_distance_between_clusters;

        return result;
    }
}
