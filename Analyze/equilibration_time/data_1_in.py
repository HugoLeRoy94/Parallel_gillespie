import numpy as np
# gillespie parameter
Nlinker = 2
ell_tot = 10**3
kdiff = 0.1
Energy = -15

Nprocess = 1000
seeds = set()
while len(seeds) < Nprocess:
    seeds.add(np.random.randint(1000000))
seeds = list(seeds)
args = [[ell_tot,Energy,kdiff,seeds[_],Nlinker,3] for _ in range(Nprocess)]

# argument of the different classes
cluster_arg = tuple([3.]) # max distance
MSD_arg = () # no argument 
ISF_arg = (0.5,10) # q_norm, q_num_sample
NRG_arg = ()
PCF_arg = (np.sqrt(ell_tot)/2,50) # max_distance,numb_bin
PCF_L_arg = (ell_tot,30) # max_distance,numb_bin

# Simulation parameters
step_tot = 2*10**4
#check_steps = 10**2
initial_check_steps = step_tot
coarse_grained_step = 10**1
log_base=False