from Cluster import Cluster
from ISF import ISF
from MSD import MSD
from Energy import NRG
from PCF import PCF
from Time import Time

import numpy as np
import multiprocessing as mp
import tables as pt
import ctypes
import sys
sys.path.append('/home/hcleroy/PostDoc/aging_condensates/Simulation/Gillespie/Gillespie_backend/')
sys.path.append('/home/hugo/PostDoc/aging_condensates/Gillespie/Gillespie_backend/')
import Gillespie_backend as gil

def compute(gillespie,output,step_tot,check_steps,coarse_grained_step,cluster_arg,MSD_arg,ISF_arg,NRG_arg,PCF_arg):
    """
    The simulation runs a total number of step_tot steps. Every measurement is first coarse grained to the number of coarse_grained steps.
    For measurement that measure the aging of time evolution measurements like MSD and ISF, check_steps correspond to the window of 
    computation
    """
    cluster = Cluster(step_tot,check_steps,coarse_grained_step,gillespie,*cluster_arg)
    isf = ISF(step_tot,check_steps,coarse_grained_step,gillespie,*ISF_arg)
    msd = MSD(step_tot,check_steps,coarse_grained_step,gillespie,*MSD_arg)
    nrg = NRG(step_tot,check_steps,coarse_grained_step,gillespie,*NRG_arg)
    pcf = PCF(step_tot,check_steps,coarse_grained_step,gillespie,*PCF_arg)
    time_track = Time(step_tot,check_steps,coarse_grained_step,gillespie)

    for i in range(step_tot//check_steps//coarse_grained_step):        
        isf.start_check_step()
        msd.start_check_step()
        pcf.start_check_step()
        time_track.start_check_step()
        for t in range(check_steps//coarse_grained_step):
            time_track.start_coarse_step()
            cluster.start_coarse_step()
            nrg.start_coarse_step()
            for steps in range(coarse_grained_step):
                move,time = gillespie.evolve()
                time_track.compute(time,move)
                cluster.compute(time,move)
                pcf.compute(time,move)
                nrg.compute(time,move)
            isf.compute(time,move,i,t)
            msd.compute(time,move,i,t)

            time_track.end_coarse_step()
            nrg.end_coarse_step()
            cluster.end_coarse_step()
        time_track.end_check_step(i,t)
        pcf.end_check_step(output,i)
        isf.end_check_step()
        msd.end_check_step()
    cluster.close(output)
    isf.close(output)
    msd.close(output)
    nrg.close(output)
    time_track.close(output)


def run_simulation(inqueue, output, step_tot, check_steps,coarse_grained_step,cluster_arg,MSD_arg,ISF_arg,NRG_arg,PCF_arg):
    """
    Run the simulation for each set of parameters fetched from the input queue.
    """
    for args in iter(inqueue.get, None):
        gillespie = initialize_gillespie(*args)
        output.put(('create_group',('/','S'+hex(gillespie.seed))))
        compute(gillespie, output, step_tot, check_steps,coarse_grained_step,cluster_arg,MSD_arg,ISF_arg,NRG_arg,PCF_arg)

    
def handle_output(output, filename, header):
    """
    Handles writing simulation results to an HDF5 file.
    """
    with pt.open_file(filename, mode='w') as hdf:
        hdf.root._v_attrs.file_header = header
        
        while True:
            task = output.get()
            if task is None: break  # Signal to terminate
            
            method, args = task
            getattr(hdf, method)(*args)

def initialize_gillespie(ell_tot, Energy, kdiff, seed, Nlinker, dimension):
    """
    Initialize the Gillespie simulation system with the given parameters.
    """
    # Assuming gil.Gillespie is the correct way to initialize your Gillespie object
    return gil.Gillespie(ell_tot=ell_tot, rho0=0., BindingEnergy=Energy, kdiff=kdiff,
                         seed=seed, sliding=False, Nlinker=Nlinker, old_gillespie=None, dimension=dimension)

def parallel_evolution(args, step_tot, check_steps,coarse_grained_step,filename,cluster_arg,MSD_arg,ISF_arg,NRG_arg,PCF_arg):
    """
    Coordinate parallel execution of MSD evolution simulations.
    """
    num_process = mp.cpu_count()
    output = mp.Queue()
    inqueue = mp.Queue()
    
    header = make_header(args, [step_tot, check_steps,coarse_grained_step,cluster_arg,MSD_arg,ISF_arg,NRG_arg,PCF_arg])
    proc = mp.Process(target=handle_output, args=(output, filename, header))
    proc.start()
    
    jobs = [mp.Process(target=run_simulation, 
                       args=(inqueue, output, step_tot, check_steps,coarse_grained_step,cluster_arg,MSD_arg,ISF_arg,NRG_arg,PCF_arg)) 
                       for _ in range(num_process)]
    
    for job in jobs:
        job.start()
    
    for arg in args:
        inqueue.put(arg)
    
    for _ in jobs:
        inqueue.put(None)
    
    for job in jobs:
        job.join()
    
    output.put(None)  # Signal to `handle_output` to terminate
    proc.join()



def make_header(args, sim_arg):
    """
    Create a header string for the HDF5 file.
    """
    header = f"This file computes the mean squared displacement. Parameters of the simulation:\n"
    labels = ['ell_tot', 'Energy', 'kdiff', 'seed', 'Nlinker', 'dimension', 'step_tot', 'check_steps','q_norm']
    values = args + sim_arg
    header += '\n'.join([f"{label} = {value}" for label, value in zip(labels, values)])
    return header
