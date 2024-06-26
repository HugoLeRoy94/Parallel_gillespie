import numpy as np
from scipy.spatial import distance_matrix
def histogram_float(*args, **kwargs):
    counts, bin_edges = np.histogram(*args, **kwargs)
    return counts.astype(float), bin_edges

class PCF:
    def __init__(self,step_tot,log_check_points,coarse_grained_steps,gillespie,max_distance,num_bins,*args):
        self.num_bins = num_bins
        self.max_distance = max_distance
        counts,self.bin_edges = histogram_float([],bins=self.num_bins,range=(0,self.max_distance))
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        self.bin_widths = self.bin_edges[1:] - self.bin_edges[:-1]
        #self.shell_volumes = (4 / 3) * np.pi * ((self.bin_centers + self.bin_widths)**3 - self.bin_centers**3)
        self.shell_volumes = 4/3 * np.pi * (self.bin_edges[1:]**3 - self.bin_edges[:-1]**3)
        self.PCF_counts = np.zeros((len(log_check_points),num_bins,2) )        
        self.gillespie = gillespie
        self.t_tot = 0.
    def start_check_step(self,*args):
        self.counts = np.zeros(self.num_bins,dtype=float)
        self.prev_hist = np.zeros(self.num_bins,dtype=float)
        self.time = 0.
    def compute(self,time,move,*args):
        self.time+=np.sum(time)
        self.counts+=self.prev_hist*time[0]
        dist_matrix = distance_matrix(self.gillespie.get_r(),self.gillespie.get_r())
        self.dist = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
        # We need to ignore the self-distances by setting the diagonal to infinity
        #np.fill_diagonal(self.dist_matrix, np.inf)
        # Find the nearest neighbor distances
        #nearest_neighbor_distances = np.min(self.dist_matrix, axis=1)
        self.prev_hist, bin_edges = histogram_float(self.dist, bins=self.num_bins, range=(0, self.max_distance))
        #self.prev_hist, _ = histogram_float(nearest_neighbor_distances, bins=self.bin_edges, range=(self.bin_edges[0], self.bin_edges[-1]))
    def end_check_step(self,i,*args):
        self.counts = self.counts / (self.time * self.shell_volumes*self.dist.shape[0])
        self.t_tot+=self.time
        self.PCF_counts[i] = np.stack((self.bin_centers,self.counts), axis=-1)
    def close(self,output,*args):
        output.put(('create_array',('/'+'S'+hex(self.gillespie.seed),'PCF',self.PCF_counts)))
class PCF_L:
    def __init__(self,step_tot,log_check_points,coarse_grained_step,gillespie,max_distance,num_bins,*args):
        self.num_bins = num_bins
        self.max_distance = max_distance
        counts,self.bin_edges = histogram_float([],bins=self.num_bins,range=(0,self.max_distance))
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        self.bin_widths = self.bin_edges[1:] - self.bin_edges[:-1]
        #self.PCF_counts = np.zeros((step_tot // check_steps,num_bins,2) )
        self.PCF_counts = np.zeros((len(log_check_points),num_bins,2) )
        self.gillespie = gillespie
        self.t_tot = 0.
    def start_check_step(self,*args):
        self.counts = np.zeros(self.num_bins,dtype=float)
        self.prev_hist = np.zeros(self.num_bins,dtype=float)
        self.time = 0.
    def compute(self,time,move,*args):
        self.time+=np.sum(time)
        self.counts+=self.prev_hist*time[0]
        self.dist = np.zeros(self.gillespie.get_ell_coordinates().shape[0]+1)        
        self.dist[1:-1]= self.gillespie.get_ell_coordinates()[1:] - self.gillespie.get_ell_coordinates()[:-1]
        self.dist[0] = self.gillespie.get_ell_coordinates()[0]
        self.dist[-1] = self.gillespie.ell_tot - self.gillespie.get_ell_coordinates()[-1]
        #dist_matrix = distance_matrix(self.gillespie.get_r(),self.gillespie.get_r())
        #self.dist = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
        self.prev_hist, _ = histogram_float(self.dist, bins=self.bin_edges, range=(self.bin_edges[0], self.bin_edges[-1]),density=False) #/ self.fist.shape[0]
        #self.prev_hist, bin_edges = histogram_float(self.dist, bins=self.num_bins, range=(0, self.max_distance))
    def end_check_step(self,i,*args):
        self.counts = self.counts / (self.time)
        self.t_tot+=self.time
        self.PCF_counts[i] = np.stack((self.bin_centers,self.counts), axis=-1)
    def close(self,output,*args):
        output.put(('create_array',('/'+'S'+hex(self.gillespie.seed),'PCF_L',self.PCF_counts)))
        