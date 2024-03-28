import numpy as np
import copy
import tables as pt

class MSD:
    def __init__(self,step_tot,log_check_points,coarse_grained_steps,gillespie,*args):
        #self.msd_time = np.zeros((step_tot // check_steps, check_steps//coarse_grained_step), dtype=float)
        self.msd_time = [np.zeros((end-start)//coarse_grained_steps, dtype=float) for start, end in zip([0]+log_check_points[:-1], log_check_points)]
        self.msd_tot = np.zeros((step_tot//coarse_grained_steps), dtype=float)
        self.gillespie = gillespie
        self.index_tot = 0
        self.sim_initial_positions = copy.copy(gillespie.get_r(periodic=True))
    def compute(self,time,move,i,t,*args):
        self.msd_time[i][t] = np.mean(np.linalg.norm(self.gillespie.get_r(periodic=True) - self.initial_positions, axis=1)**2)
        self.msd_tot[self.index_tot] = np.mean(np.linalg.norm(self.gillespie.get_r(periodic=True) - self.sim_initial_positions, axis=1)**2)
        self.index_tot+=1
    def start_check_step(self,*args):
        self.initial_positions = copy.copy(self.gillespie.get_r(periodic=True))
    def end_check_step(self,*args):
        return
    def close(self,output,*args):
        #output.put(('create_vlarray', ('/'+'S'+hex(self.gillespie.seed),'MSD' , self.msd_time)))
        output.put(('create_array', ('/'+'S'+hex(self.gillespie.seed),'MSD_tot' , self.msd_tot)))
        output.put(('create_vlarray', ('/S'+hex(self.gillespie.seed), 'MSD', pt.Float64Atom(shape=()), self.msd_time)))
        #output.put(('create_vlarray', ('/S'+hex(self.gillespie.seed), 'MSD_tot', pt.Float64Atom(shape=()), self.time_check_steps)))
