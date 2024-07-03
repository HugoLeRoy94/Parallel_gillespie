import numpy as np


class Picture:
    def __init__(self,step_tot,check_steps,coarse_grained_step,gillespie,*args):
        self.gillespie= gillespie
        self.pictures = list()
    def compute(self,time,move,*args):
        return
    def start_check_step(self,*args):
        return
    def end_check_step(self,*args):
        self.pictures.append(self.gillespie.get_r())
    def close(self,output,*args):
        self.pictures = np.array(self.pictures)
        output.put(('create_array', ('/'+'S'+hex(self.gillespie.seed),'Picture' , self.pictures)))
