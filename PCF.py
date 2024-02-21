import numpy as np
from scipy.spatial import distance_matrix
def histogram_float(*args, **kwargs):
    counts, bin_edges = np.histogram(*args, **kwargs)
    return counts.astype(float), bin_edges
class PCF:
    def __init__(self,step_tot,check_steps,coarse_grained_step,gillespie,max_distance,num_bins):
        self.num_bins = num_bins
        self.max_distance = max_distance
        counts,bin_edges = histogram_float([],bins=self.num_bins,range=(0,self.max_distance))
        self.bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        self.bin_widths = bin_edges[1:] - bin_edges[:-1]
        self.shell_volumes = (4 / 3) * np.pi * ((self.bin_centers + self.bin_widths)**3 - self.bin_centers**3)
        self.gillespie = gillespie
        self.t_tot = 0.
    def start_check_step(self):
        self.counts, bin_edges = histogram_float([], bins=self.num_bins, range=(0, self.max_distance))
        self.prev_hist = np.zeros(self.counts.shape,dtype=float)
        self.time = 0.
    def compute(self,time,move):
        self.time+=np.sum(time)
        self.counts+=self.prev_hist*time[0]
        dist_matrix = distance_matrix(self.gillespie.get_r(),self.gillespie.get_r())
        self.dist = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
        self.prev_hist, bin_edges = histogram_float(self.dist, bins=self.num_bins, range=(0, self.max_distance))
    def end_check_step(self,output,i):
        self.counts = self.counts / (self.time * self.shell_volumes*self.dist.shape[0])
        self.t_tot+=self.time
        output.put(('create_array',('/'+'S'+hex(self.gillespie.seed),'PCF_'+str(i),np.stack((self.bin_centers,self.counts), axis=-1))))
        