import numpy as np
from scipy.spatial import distance_matrix
def histogram_float(*args, **kwargs):
    counts, bin_edges = np.histogram(*args, **kwargs)
    return counts.astype(float), bin_edges
class PCF:
    def __init__(self,step_tot,check_steps,coarse_grained_step,gillespie,max_distance,num_bins,LOG=False):
        self.num_bins = num_bins
        self.max_distance = max_distance
        self.LOG = LOG
        if LOG:
            min_distance = 0.1  # Example: set based on your system's smallest meaningful distance
            self.bin_edges = np.logspace(np.log10(min_distance), np.log10(self.max_distance), self.num_bins + 1)        
            self.bin_centers = 10**((np.log10(self.bin_edges[:-1]) + np.log10(self.bin_edges[1:])) / 2)
            self.bin_widths = np.diff(self.bin_edges)
        else:
            counts,self.bin_edges = histogram_float([],bins=self.num_bins,range=(0,self.max_distance))
            self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
            self.bin_widths = self.bin_edges[1:] - self.bin_edges[:-1]
        #self.shell_volumes = (4 / 3) * np.pi * ((self.bin_centers + self.bin_widths)**3 - self.bin_centers**3)
        self.shell_volumes = 4/3 * np.pi * (self.bin_edges[1:]**3 - self.bin_edges[:-1]**3)
        self.PCF_counts = np.zeros((step_tot // check_steps,num_bins,2) )
        self.gillespie = gillespie
        self.t_tot = 0.
    def start_check_step(self):
        self.counts = np.zeros(self.num_bins,dtype=float)
        self.prev_hist = np.zeros(self.num_bins,dtype=float)
        self.time = 0.
    def compute(self,time,move):
        self.time+=np.sum(time)
        self.counts+=self.prev_hist*time[0]
        dist_matrix = distance_matrix(self.gillespie.get_r(),self.gillespie.get_r())
        self.dist = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
        #self.prev_hist, bin_edges = histogram_float(self.dist, bins=self.num_bins, range=(0, self.max_distance))
        self.prev_hist, _ = histogram_float(self.dist, bins=self.bin_edges, range=(self.bin_edges[0], self.bin_edges[-1]))
    def end_check_step(self,i):
        self.counts = self.counts / (self.time * self.shell_volumes*self.dist.shape[0])
        self.t_tot+=self.time
        self.PCF_counts[i] = np.stack((self.bin_centers,self.counts), axis=-1)
    def close(self,output):
        output.put(('create_array',('/'+'S'+hex(self.gillespie.seed),'PCF',self.PCF_counts)))
class PCF_L:
    def __init__(self,step_tot,check_steps,coarse_grained_step,gillespie,max_distance,num_bins,LOG=True):
        self.num_bins = num_bins
        self.max_distance = max_distance
        if LOG:
            min_distance = 1.  # Example: set based on your system's smallest meaningful distance
            self.bin_edges = np.logspace(np.log10(min_distance), np.log10(self.max_distance), self.num_bins + 1)        
            self.bin_centers = 10**((np.log10(self.bin_edges[:-1]) + np.log10(self.bin_edges[1:])) / 2)
            self.bin_widths = np.diff(self.bin_edges)
        else:        
            counts,self.bin_edges = histogram_float([],bins=self.num_bins,range=(0,self.max_distance))
            self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
            self.bin_widths = self.bin_edges[1:] - self.bin_edges[:-1]
        self.PCF_counts = np.zeros((step_tot // check_steps,num_bins,2) )
        self.shell_volumes = (4 / 3) * np.pi * ((self.bin_centers + self.bin_widths)**3 - self.bin_centers**3)
        self.gillespie = gillespie
        self.t_tot = 0.
    def start_check_step(self):
        self.counts = np.zeros(self.num_bins,dtype=float)
        self.prev_hist = np.zeros(self.num_bins,dtype=float)
        self.time = 0.
    def compute(self,time,move):
        self.time+=np.sum(time)
        self.counts+=self.prev_hist*time[0]
        self.dist = self.gillespie.get_ell_coordinates()[1:] - self.gillespie.get_ell_coordinates()[:-1]
        #dist_matrix = distance_matrix(self.gillespie.get_r(),self.gillespie.get_r())
        #self.dist = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
        self.prev_hist, _ = histogram_float(self.dist, bins=self.bin_edges, range=(self.bin_edges[0], self.bin_edges[-1]))
        #self.prev_hist, bin_edges = histogram_float(self.dist, bins=self.num_bins, range=(0, self.max_distance))
    def end_check_step(self,i):
        self.counts = self.counts / (self.time * self.shell_volumes*self.dist.shape[0])
        self.t_tot+=self.time
        self.PCF_counts[i] = np.stack((self.bin_centers,self.counts), axis=-1)
    def close(self,output):
        output.put(('create_array',('/'+'S'+hex(self.gillespie.seed),'PCF_L',self.PCF_counts)))
        