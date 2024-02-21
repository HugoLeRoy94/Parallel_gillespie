import numpy as np
import copy

class MSD:
    def __init__(self,step_tot,check_steps,coarse_grained_step,gillespie):
        self.msd_time = np.zeros((step_tot // check_steps//coarse_grained_step, check_steps//coarse_grained_step, 3), dtype=float)
        self.msd_tot = np.zeros((step_tot//coarse_grained_step, 2), dtype=float)
        self.gillespie = gillespie
        self.index_tot = 0
        self.sim_initial_positions = copy.copy(gillespie.get_r(periodic=True))
    def compute(self,time,move,i,t):
        self.msd_time[i,t] = np.mean(np.linalg.norm(self.gillespie.get_r(periodic=True) - self.initial_positions, axis=1)**2)
        self.msd_tot[self.index_tot] = np.mean(np.linalg.norm(self.gillespie.get_r(periodic=True) - self.sim_initial_positions, axis=1)**2)
        self.index_tot+=1
    def start_check_step(self):
        self.initial_positions = copy.copy(self.gillespie.get_r(periodic=True))
    def end_check_step(self):
        return
    def close(self,output):
        output.put(('create_array', ('/'+'S'+hex(self.gillespie.seed),'MSD' , self.msd_time)))
        output.put(('create_array', ('/'+'S'+hex(self.gillespie.seed),'MSD_tot' , self.msd_tot)))