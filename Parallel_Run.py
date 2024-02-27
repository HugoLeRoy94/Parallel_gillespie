from Cluster import Cluster
from ISF import ISF
from MSD import MSD
from Energy import NRG
from PCF import PCF
from PCF import PCF_L
from Time import Time

import numpy as np
import multiprocessing as mp
import tables as pt
import ctypes
import sys
sys.path.append('/home/hcleroy/PostDoc/aging_condensates/Simulation/Gillespie/Gillespie_backend/')
sys.path.append('/home/hcleroy/aging_condensate/Gillespie/Gillespie_backend')
sys.path.append('/home/hugo/PostDoc/aging_condensates/Gillespie/Gillespie_backend/')

import Gillespie_backend as gil

def compute(gillespie, output, step_tot, initial_check_steps, coarse_grained_step, cluster_arg, MSD_arg, ISF_arg, NRG_arg, PCF_arg, PCF_L_arg, log_base=2):
    """
    Implements logarithmic sampling for measurements in a Gillespie simulation.
    initial_check_steps is the initial number of steps between checks, and log_base determines the base of the logarithmic scale for step increment.
    """
    # Calculate logarithmic check points
    max_exponent = np.log(step_tot / initial_check_steps) / np.log(log_base)
    log_check_points = [int(initial_check_steps * log_base ** i) for i in range(int(max_exponent) + 1)]
    log_check_points.append(step_tot)
    
    cluster = Cluster(step_tot, log_check_points, coarse_grained_step, gillespie, *cluster_arg)
    isf = ISF(step_tot, log_check_points, coarse_grained_step, gillespie, *ISF_arg)
    msd = MSD(step_tot, log_check_points, coarse_grained_step, gillespie, *MSD_arg)
    nrg = NRG(step_tot, log_check_points, coarse_grained_step, gillespie, *NRG_arg)
    pcf = PCF(step_tot, log_check_points, coarse_grained_step, gillespie, *PCF_arg)
    pcf_L = PCF_L(step_tot, log_check_points, coarse_grained_step, gillespie, *PCF_L_arg)
    time_track = Time(step_tot, log_check_points, coarse_grained_step, gillespie)

    current_step = 0
    for i,check_point in enumerate(log_check_points):
        #while current_step < check_point:
        isf.start_check_step()
        msd.start_check_step()
        pcf.start_check_step()
        pcf_L.start_check_step()
        time_track.start_check_step(i)
        # Adjust the inner loop to match the current checkpoint interval
        steps_to_next_checkpoint = check_point - current_step
        for t in range(steps_to_next_checkpoint // coarse_grained_step):
            cluster.start_coarse_step()
            nrg.start_coarse_step()
            for steps in range(coarse_grained_step):
                move, time = gillespie.evolve()
                time_track.compute(time, move)
                cluster.compute(time, move)
                pcf.compute(time, move)
                pcf_L.compute(time, move)
                nrg.compute(time, move)
            isf.compute(time, move, i, t)
            msd.compute(time, move,i, t)
            time_track.end_coarse_step()
            nrg.end_coarse_step()
            cluster.end_coarse_step()
            time_track.end_check_step(i, t)
        current_step += steps_to_next_checkpoint
        pcf.end_check_step(i)
        pcf_L.end_check_step(i)
        isf.end_check_step()
        msd.end_check_step()
    
    pcf.close(output)
    pcf_L.close(output)
    cluster.close(output)
    isf.close(output)
    msd.close(output)
    nrg.close(output)
    time_track.close(output)

#def compute(gillespie,output,step_tot,check_steps,coarse_grained_step,cluster_arg,MSD_arg,ISF_arg,NRG_arg,PCF_arg,PCF_L_arg):
#    """
#    The simulation runs a total number of step_tot steps. Every measurement is first coarse grained to the number of coarse_grained steps.
#    For measurement that measure the aging of time evolution measurements like MSD and ISF, check_steps correspond to the window of 
#    computation
#    """
#    cluster = Cluster(step_tot,check_steps,coarse_grained_step,gillespie,*cluster_arg)
#    isf = ISF(step_tot,check_steps,coarse_grained_step,gillespie,*ISF_arg)
#    msd = MSD(step_tot,check_steps,coarse_grained_step,gillespie,*MSD_arg)
#    nrg = NRG(step_tot,check_steps,coarse_grained_step,gillespie,*NRG_arg)
#    pcf = PCF(step_tot,check_steps,coarse_grained_step,gillespie,*PCF_arg)
#    pcf_L = PCF_L(step_tot,check_steps,coarse_grained_step,gillespie,*PCF_L_arg)
#    time_track = Time(step_tot,check_steps,coarse_grained_step,gillespie)
#
#    for i in range(step_tot//check_steps):
#        isf.start_check_step()
#        msd.start_check_step()
#        pcf.start_check_step()
#        pcf_L.start_check_step()
#        time_track.start_check_step(i)
#        for t in range(check_steps//coarse_grained_step):
#            #time_track.start_coarse_step()
#            cluster.start_coarse_step()
#            nrg.start_coarse_step()
#            for steps in range(coarse_grained_step):
#                move,time = gillespie.evolve()
#                time_track.compute(time,move)
#                cluster.compute(time,move)
#                pcf.compute(time,move)
#                pcf_L.compute(time,move)
#                nrg.compute(time,move)
#            isf.compute(time,move,i,t)
#            msd.compute(time,move,i,t)
#
#            time_track.end_coarse_step()
#            nrg.end_coarse_step()
#            cluster.end_coarse_step()
#            time_track.end_check_step(i,t)
#        pcf.end_check_step(i)
#        pcf_L.end_check_step(i)
#        isf.end_check_step()
#        msd.end_check_step()
#    pcf.close(output)
#    pcf_L.close(output)
#    cluster.close(output)
#    isf.close(output)
#    msd.close(output)
#    nrg.close(output)
#    time_track.close(output)


def run_simulation(inqueue, output, step_tot, check_steps,coarse_grained_step,cluster_arg,MSD_arg,ISF_arg,NRG_arg,PCF_arg,PCF_L_arg):
    """
    Run the simulation for each set of parameters fetched from the input queue.
    """
    for args in iter(inqueue.get, None):
        gillespie = initialize_gillespie(*args)
        output.put(('create_group',('/','S'+hex(gillespie.seed))))
        compute(gillespie, output, step_tot, check_steps,coarse_grained_step,cluster_arg,MSD_arg,ISF_arg,NRG_arg,PCF_arg,PCF_L_arg)

    
#def handle_output(output, filename, header):
#    """
#    Handles writing simulation results to an HDF5 file.
#    """
#    with pt.open_file(filename, mode='w') as hdf:
#        hdf.root._v_attrs.file_header = header
#        
#        while True:
#            task = output.get()
#            if task is None: break  # Signal to terminate
#            
#            method, args = task
#            getattr(hdf, method)(*args)
def handle_output(output, filename, header):
    """
    Handles writing simulation results to an HDF5 file, including support for
    variable-length arrays via the 'create_vlarray' command.
    """
    with pt.open_file(filename, mode='w') as hdf:
        hdf.root._v_attrs.file_header = header
        
        while True:
            task = output.get()
            if task is None:
                break  # Signal to terminate
            
            method, args = task
            
            # Handling different types of tasks
            if method == 'create_vlarray':
                # Create a variable-length array
                group_path, array_name, atom, data = args
                vlarray = hdf.create_vlarray(group_path, array_name, atom)
                for item in data:
                    vlarray.append(item)
            else:# method == 'create_array':
                # Create a fixed-size array
                getattr(hdf, method)(*args)
                #group_path, array_name, data = args
                #hdf.create_array(group_path, array_name, obj=data) 

def initialize_gillespie(ell_tot, Energy, kdiff, seed, Nlinker, dimension):
    """
    Initialize the Gillespie simulation system with the given parameters.
    """
    # Assuming gil.Gillespie is the correct way to initialize your Gillespie object
    return gil.Gillespie(ell_tot=ell_tot, rho0=0., BindingEnergy=Energy, kdiff=kdiff,
                         seed=seed, sliding=False, Nlinker=Nlinker, old_gillespie=None, dimension=dimension)

def parallel_evolution(args, step_tot, check_steps,coarse_grained_step,filename,cluster_arg,MSD_arg,ISF_arg,NRG_arg,PCF_arg,PCF_L_arg):
    """
    Coordinate parallel execution of MSD evolution simulations.
    """
    num_process = mp.cpu_count()
    output = mp.Queue()
    inqueue = mp.Queue()
    
    header = make_header(args, [step_tot, check_steps,coarse_grained_step,cluster_arg,MSD_arg,ISF_arg,NRG_arg,PCF_arg,PCF_L_arg])
    proc = mp.Process(target=handle_output, args=(output, filename, header))
    proc.start()
    
    jobs = [mp.Process(target=run_simulation, 
                       args=(inqueue, output, step_tot, check_steps,coarse_grained_step,cluster_arg,MSD_arg,ISF_arg,NRG_arg,PCF_arg,PCF_L_arg)) 
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
    Create a header string for the HDF5 file, correctly indicating the value of the different variables.
    """
    header = "This HDF5 file contains the results of a  Gillespie simulation, focusing on properties such as (MSD), (ISF),  (NRG), (PCF) alongside clustering behavior. Each simulation run is uniquely identified by a hexadecimal seed value,\n"
    header+="How to Navigate the Data:\n"
    header+="Groups: Each top-level group in this file is named after a seed value in hexadecimal format (0x...) representing the seed.\n"
    header+="Datasets within Groups: Within each group, you'll find datasets corresponding to different measurements taken during the simulation, such as MSD, ISF, NRG, PCF, and cluster data. These are labeled according to the type of measurement.\n"
    header+="Reading Data: To access the data, navigate to the desired group (seed) and then select the measurement dataset within that group. The data format and structure may vary by measurement type, typically represented in multidimensional arrays where dimensions correspond to simulation time steps, spatial dimensions, or specific parameters like wavevector magnitudes for ISF.\n"
    header+="This dataset is designed for comprehensive analysis of the modeled system, allowing for deep dives into both its dynamic and structural properties. For further details on the simulation parameters and setup, refer to the header section at the beginning of this file.:\n"

    # Assuming the first element in args is representative for all Gillespie parameters
    if args:
        first_set_args = args[0]  # Taking the first set of parameters as representative
        labels_gillespie = ['ell_tot', 'Energy', 'kdiff', 'seed', 'Nlinker', 'dimension']
        header += '\n'.join([f"{label} = {value}" for label, value in zip(labels_gillespie, first_set_args)])

    # Adding simulation-wide parameters from sim_arg
    labels_sim = ['step_tot', 'check_steps', 'coarse_grained_step', 'cluster_max_distance', 'MSD_args', 'ISF_q_norm', 'ISF_q_num_sample', 'NRG_args', 'PCF_max_distance', 'PCF_num_bins','PCF_L_max_distance','PCF_L_num_bins']
    
    # Ensure sim_arg is unpacked correctly according to how you structure it.
    # This example directly uses sim_arg assuming it is in the correct order matching labels_sim.
    header += '\n' + '\n'.join([f"{label} = {value}" for label, value in zip(labels_sim, sim_arg)])

    return header

