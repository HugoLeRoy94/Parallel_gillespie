import numpy as np


class NRG:
    def __init__(self,step_tot,check_steps,coarse_grained_step,gillespie,*args):
        self.gillespie= gillespie
        self.Energy = np.zeros(step_tot//coarse_grained_step,dtype=float)
        self.Entropy = np.zeros(step_tot//coarse_grained_step,dtype=float)
        self.index = 0
        #self.prev_energy = gillespie.get_F()
        self.Eb = self.gillespie.binding_energy
        self.prev_energy = (self.gillespie.get_N_loop()-1)*self.Eb
        self.prev_entropy = gillespie.get_S()/(self.gillespie.get_N_loop()-1)
        self.move1_count = 0
        self.step_tot = step_tot
        
    def compute(self,time,move,*args):
        dt = np.sum(time)
        self.t_tot+=dt
        self.Energy[self.index]+=self.prev_energy*dt
        self.Entropy[self.index]+=self.prev_entropy*dt
        #self.prev_energy = self.gillespie.get_F()
        self.prev_energy = (self.gillespie.get_N_loop()-1)*self.Eb
        self.prev_entropy = (self.gillespie.get_S())
        self.move1_count+=np.count_nonzero(move == 1)
    def start_coarse_step(self,*args):
        self.t_tot=0.
    def end_coarse_step(self,*args):
        self.Energy[self.index]/=self.t_tot
        self.Entropy[self.index]/=self.t_tot
        self.index+=1
    def close(self,output,*args):
        output.put(('create_array', ('/'+'S'+hex(self.gillespie.seed),'NRG' , self.Energy)))
        output.put(('create_array', ('/'+'S'+hex(self.gillespie.seed),'Entropy' , self.Entropy)))
        output.put(('create_array',('/'+'S'+hex(self.gillespie.seed),'moves_1' , np.array([self.move1_count/self.step_tot]))))