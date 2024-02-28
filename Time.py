import numpy as np
import tables as pt

class Time:
    def __init__(self,step_tot,log_check_points,coarse_grained_steps,gillespie):
        self.gillespie = gillespie
        self.total_coarse_grained_time = np.zeros(step_tot//coarse_grained_steps,dtype=float)
        #self.time_check_steps = np.zeros((len(log_check_points),check_steps//coarse_grained_steps),dtype=float)
        self.time_check_steps = [np.zeros((end-start)//coarse_grained_steps, dtype=float) for start, end in zip([0]+log_check_points[:-1], log_check_points)]
        self.time_shift = np.zeros(len(log_check_points),dtype=float)
        self.index = 0
        self.time_coarse = 0
    def compute(self,time,move):
        self.time_coarse+=np.sum(time)
        self.time_check+=np.sum(time)
    def start_check_step(self,i):
        self.time_check = 0.
        self.time_shift[i] = self.time_coarse
    #def start_coarse_step(self):
    #    self.time_coarse =0.
    def end_check_step(self,i,t):
        self.time_check_steps[i][t] = self.time_check
    def end_coarse_step(self):
        self.total_coarse_grained_time[self.index] = self.time_coarse
        self.index+=1
    def close(self,output):
        output.put(('create_array',('/S'+hex(self.gillespie.seed),'Coarse_Time',self.total_coarse_grained_time)))
        #output.put(('create_vl  array',('/S'+hex(self.gillespie.seed),'Check_Time',self.time_check_steps)))
        output.put(('create_vlarray', ('/S'+hex(self.gillespie.seed), 'Check_Time', pt.Float64Atom(shape=()), self.time_check_steps)))
        output.put(('create_array',('/S'+hex(self.gillespie.seed),'Time_shift',self.time_shift)))
        