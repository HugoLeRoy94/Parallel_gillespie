import numpy as np


class NRG:
    def __init__(self,step_tot,check_steps,coarse_grained_step,gillespie):
        self.Energy = np.zeros(step_tot//coarse_grained_step,dtype=float)
        self.index = 0
        self.prev_energy = gillespie.get_F()
        self.move1_count = 0
        self.step_tot = step_tot
        self.gillespie= gillespie
    def compute(self,time,move):
        dt = np.sum(time)
        self.t_tot+=dt
        self.Energy[self.index]+=self.prev_energy*dt
        self.prev_energy = self.gillespie.get_F()
        self.move1_count+=np.count_nonzero(move == 1)
    def start_coarse_step(self):
        self.t_tot=0.        
    def end_coarse_step(self):
        self.Energy[self.index]/=self.t_tot
        self.index+=1
    def close(self,output):
        output.put(('create_array', ('/'+'S'+hex(self.gillespie.seed),'NRG' , self.Energy)))
        output.put(('create_array',('/'+'S'+hex(self.gillespie.seed),'moves_1' , np.array([self.move1_count/self.step_tot]))))