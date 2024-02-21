import numpy as np
from scipy.spatial import distance_matrix

def cluster_points(points, max_distance):
    # Reshape the points array to shape (-1, 3) if it's (3, N)
    if points.shape[0] == 3:
        points = points.T

    N = points.shape[0]
    clusters = []
    for i in range(N):
        # Initialize a new cluster with the current point
        new_cluster = [points[i]]

        for cluster in clusters:
            for point in cluster:
                # If the current point is within the max_distance of a point in an existing cluster
                if np.linalg.norm(points[i]- point) <= max_distance:
                    # Add the current point to the cluster and break out of the loop
                    cluster.append(points[i])
                    break
            else:
                # Continue the loop if the inner loop wasn't broken
                continue

            # Break the outer loop if the inner loop was broken
            break
        else:
            # If the point wasn't added to any cluster, we add the new_cluster to the list of clusters
            clusters.append(new_cluster)

    return [np.array(cluster) for cluster in clusters]

def compute_mean_distance_between_clusters(clusters):
    """Compute the mean distance between cluster centroids."""
    if len(clusters) < 2:  # If there's one or no cluster, mean distance isn't applicable
        return np.nan
    centroids = [np.mean(cluster, axis=0) for cluster in clusters] # center of the cluster
    pairwise_distances = distance_matrix(centroids, centroids)
    np.fill_diagonal(pairwise_distances, np.nan)  # Ignore self-distances
    mean_distance = np.nanmean(pairwise_distances)
    return mean_distance


class Cluster:
    def __init__(self,step_tot,check_steps,coarse_grained_step,gillespie,max_distance):
        self.metrics_time = np.zeros((step_tot //coarse_grained_step, 2), dtype=float)  # Adjusted for an extra column
        clusters = cluster_points(gillespie.get_r(), max_distance)
        self.prev_c_size = np.mean([len(c) for c in clusters])
        self.prev_mean_distance = compute_mean_distance_between_clusters(clusters)
        self.gillespie = gillespie
        self.max_distance = max_distance
        self.index = 0
    def compute(self,time,move):
        dt = np.sum(time)
        self.t_tot+=dt
        clusters = cluster_points(self.gillespie.get_R(), self.max_distance)
        c_size = np.mean([len(c) for c in clusters])
        mean_distance = compute_mean_distance_between_clusters(clusters)
        self.av_c_size += self.prev_c_size * dt
        self.total_mean_distance += self.prev_mean_distance * dt if not np.isnan(self.prev_mean_distance) else 0
        self.prev_c_size = c_size
        self.prev_mean_distance = mean_distance
    def start_coarse_step(self):
        self.av_c_size = 0.
        self.total_mean_distance = 0.
        self.t_tot= 0.
    def end_coarse_step(self):
        self.av_c_size /= self.t_tot
        mean_distance_avg = self.total_mean_distance / self.t_tot if self.t_tot != 0 else np.nan
        self.metrics_time[self.index] = [self.av_c_size, mean_distance_avg]
        self.index+=1
    def close(self,output):
        output.put(('create_array', ('/'+'S'+hex(self.gillespie.seed),'cluster' , self.metrics_time)))