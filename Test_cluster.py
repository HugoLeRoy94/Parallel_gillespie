import ctypes
import os
import numpy as np

# Load the shared library
lib_path = os.path.abspath("clustering.so")
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

# Create a sample set of points in Python
points = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
])

# Flatten the points array to pass to C++
flattened_points = points.flatten().astype(ctypes.c_float)
max_dist = 10.
# Call the C++ function
stats = my_clustering_lib.hierarchical_clustering_with_stats(flattened_points.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), points.shape[0], max_dist)

print("Average Cluster Size:", stats.average_cluster_size)
print("Average Distance Between Clusters:", stats.average_distance_between_clusters)
