import numpy as np
from scipy.spatial import distance_matrix
from scipy.spatial import distance
import os
import ctypes
import pathlib
# Load the shared library
lib_path = os.path.abspath(str(pathlib.Path(__file__).parent.absolute())+"/clustering.so")
my_clustering_lib = ctypes.CDLL(lib_path)

# Define the ClusterStats structure
class ClusterStats(ctypes.Structure):
    _fields_ = [
        ("average_cluster_size", ctypes.c_float),
        ("average_distance_between_clusters", ctypes.c_float),
    ]

# Set the argument and return types for the C++ function
my_clustering_lib.hierarchical_clustering_with_stats.restype = ClusterStats
my_clustering_lib.hierarchical_clustering_with_stats.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_float]

#### Written by chat gpt : need to be tested
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances

def hierarchical_clustering_with_stats(points, max_distance):
    # Check if input is in the format (3, N), if so transpose it
    if points.shape[0] == 3:
        points = points.T

    # Setting up the Agglomerative Clustering with single linkage and a distance threshold
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=max_distance, linkage='single')
    clustering.fit(points)

    # Extract the labels and calculate the number of points in each cluster
    labels = clustering.labels_
    unique_labels = np.unique(labels)
    cluster_sizes = [np.sum(labels == label) for label in unique_labels]
    average_cluster_size = np.mean(cluster_sizes)

    # Calculate the centroids of each cluster
    centroids = np.array([points[labels == label].mean(axis=0) for label in unique_labels])
    
    # Calculate pairwise distances between centroids if more than one cluster exists
    if len(centroids) > 1:
        distances = pairwise_distances(centroids)
        # We use np.triu_indices to consider only the upper triangle of the distance matrix, excluding the diagonal
        upper_triangle_indices = np.triu_indices_from(distances, k=1)
        average_distance_between_clusters = np.mean(distances[upper_triangle_indices])
    else:
        average_distance_between_clusters = 0  # Only one cluster, no distances to compute

    return average_cluster_size, average_distance_between_clusters

def cluster_points(points, max_distance):
    # Reshape the points array to shape (-1, 3) if it's (3, N)
    if points.shape[0] == 3:
        points = points.T

    N = points.shape[0]
    clusters = []
    for i in range(N):
        # Initialize a variable to track if the point was added to any cluster
        added_to_cluster = False
        for cluster in clusters:
            for point in cluster:
                # If the current point is within the max_distance of a point in an existing cluster
                if np.linalg.norm(points[i] - point) <= max_distance:
                    # Add the current point to the cluster
                    cluster.append(points[i])
                    added_to_cluster = True
                    break  # Only break from the innermost loop
            if added_to_cluster:
                # No need to continue the inner loop if already added to this cluster
                continue

        # If the point wasn't added to any cluster, create a new cluster with this point
        if not added_to_cluster:
            clusters.append([points[i]])

    # Optional: Post-processing to merge overlapping clusters
    merged_clusters = []
    clusters_as_sets = [set(tuple(point) for point in cluster) for cluster in clusters] 
    while clusters_as_sets:        
        first, rest = clusters_as_sets[0], clusters_as_sets[1:]#clusters[0], clusters[1:]
        #first = set(tuple(x) for x in first)  # Convert lists to sets of tuples for immutability
        merged = True
        while merged:
            merged = False
            for r in rest:
                #r_set = set(tuple(x) for x in r)
                if not first.isdisjoint(r):  # Check if clusters overlap
                    first |= r  # Union the sets
                    rest.remove(r)
                    merged = True
                    break
        merged_clusters.append(list(first))
        clusters_as_sets = rest

    # Convert each cluster back to numpy arrays
    return [np.array([list(point) for point in cluster]) for cluster in merged_clusters]#[np.array(list(map(list, cluster))) for cluster in merged_clusters]

def c_size_distance(points,max_distance):
    # Flatten the points array to pass to C++
    flattened_points = points.flatten().astype(ctypes.c_float)

    # Call the C++ function
    stats = my_clustering_lib.hierarchical_clustering_with_stats(flattened_points.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), points.shape[0], max_distance)
    return stats.average_cluster_size, stats.average_distance_between_clusters

#def cluster_points(points, max_distance):
#    raise NotImplemented('the function inputs depends on the order of points')
#
#    # Reshape the points array to shape (-1, 3) if it's (3, N)
#    if points.shape[0] == 3:
#        points = points.T
#
#    N = points.shape[0]
#    clusters = []
#    for i in range(N):
#        # Initialize a new cluster with the current point
#        new_cluster = [points[i]]
#
#        for cluster in clusters:
#            for point in cluster:
#                # If the current point is within the max_distance of a point in an existing cluster
#                if np.linalg.norm(points[i]- point) <= max_distance:
#                    # Add the current point to the cluster and break out of the loop
#                    cluster.append(points[i])
#                    break
#            else:
#                # Continue the loop if the inner loop wasn't broken
#                continue
#
#            # Break the outer loop if the inner loop was broken
#            break
#        else:
#            # If the point wasn't added to any cluster, we add the new_cluster to the list of clusters
#            clusters.append(new_cluster)
#
#    return [np.array(cluster) for cluster in clusters]

def compute_mean_distance_between_clusters(clusters):
    """Compute the mean distance between cluster centroids."""
    if len(clusters) < 2:  # If there's one or no cluster, mean distance isn't applicable
        return np.nan
    centroids = [np.mean(cluster, axis=0) for cluster in clusters] # center of the cluster
    pairwise_distances = distance_matrix(centroids, centroids)
    np.fill_diagonal(pairwise_distances, np.nan)  # Ignore self-distances
    mean_distance = np.nanmean(pairwise_distances)
    return mean_distance

def compute_avg_nearest_neighbor_distance(points):
    """
    Compute the average nearest neighbor distance across the entire system.
    Parameters:
    - points: A numpy array of shape (N, D) where N is the number of points and D is the dimensionality.
    Returns:
    - The average nearest neighbor distance across all points in the system.
    """
    if len(points) < 2:
        # Not enough points to compute distances
        return np.nan 
    # Compute the pairwise distances between points
    pairwise_dist = distance.pdist(points)
    # Convert the condensed distance matrix to a square format
    square_dist = distance.squareform(pairwise_dist)
    # Replace the diagonal (self-distances) with np.inf to ignore them
    np.fill_diagonal(square_dist, np.inf)
    # Find the nearest neighbor distance for each point
    nearest_neighbor_distances = np.min(square_dist, axis=1)
    # Compute and return the average nearest neighbor distance
    return np.mean(nearest_neighbor_distances)


class Cluster:
    def __init__(self,step_tot,check_steps,coarse_grained_step,gillespie,max_distance,*args):
        self.metrics_time = np.zeros((step_tot //coarse_grained_step, 3), dtype=float)  # Adjusted for an extra column
        #clusters = cluster_points(gillespie.get_r(), max_distance)
        #self.prev_c_size = np.mean([len(c) for c in clusters])
        #self.prev_mean_distance = compute_mean_distance_between_clusters(clusters)
        #self.prev_c_size,self.prev_mean_distance =hierarchical_clustering_with_stats(gillespie.get_r(),max_distance)
        self.prev_c_size,self.prev_mean_distance =  c_size_distance(gillespie.get_r(),max_distance)
        self.prev_avg_nn_distance = compute_avg_nearest_neighbor_distance(gillespie.get_r())  # New metric
        self.gillespie = gillespie
        self.max_distance = max_distance
        self.index = 0
    def compute(self,time,move,*args):
        dt = np.sum(time)
        self.t_tot+=dt
        #clusters = cluster_points(self.gillespie.get_R(), self.max_distance)
        #c_size = np.mean([len(c) for c in clusters])
        #mean_distance = compute_mean_distance_between_clusters(clusters)
        #c_size,mean_distance = hierarchical_clustering_with_stats(self.gillespie.get_r(),self.max_distance)
        c_size,mean_distance = c_size_distance(self.gillespie.get_r(),self.max_distance)
        avg_nn_distance = compute_avg_nearest_neighbor_distance(self.gillespie.get_r())
        self.av_c_size += self.prev_c_size * dt
        self.total_mean_distance += self.prev_mean_distance * dt if not np.isnan(self.prev_mean_distance) else 0
        self.total_avg_nn_distance += self.prev_avg_nn_distance * dt if not np.isnan(self.prev_avg_nn_distance) else 0  # Accumulate new metric
        self.prev_c_size = c_size
        self.prev_mean_distance = mean_distance
        self.prev_avg_nn_distance = avg_nn_distance
    def start_coarse_step(self,*args):
        self.av_c_size = 0.
        self.total_mean_distance = 0.
        self.total_avg_nn_distance = 0.  # Reset new metric accumulator
        self.t_tot= 0.
    def end_coarse_step(self,*args):
        self.av_c_size /= self.t_tot
        mean_distance_avg = self.total_mean_distance / self.t_tot if self.t_tot != 0 else np.nan
        avg_nn_distance_avg = self.total_avg_nn_distance / self.t_tot if self.t_tot != 0 else np.nan  # Compute average for new metric
        self.metrics_time[self.index] = [self.av_c_size, mean_distance_avg, avg_nn_distance_avg]  # Store new metric        
        self.index+=1
    def close(self,output,*args):
        output.put(('create_array', ('/'+'S'+hex(self.gillespie.seed),'cluster' , self.metrics_time)))