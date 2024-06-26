import numpy as np
import copy
import tables as pt
def uniform_sphere_samples(num_samples):
        phi = np.linspace(0, 2 * np.pi, num_samples)
        cos_theta = np.linspace(-1, 1, num_samples)
        theta = np.arccos(cos_theta)
        phi, theta = np.meshgrid(phi, theta)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
        return points    
def Compute_Average_ISF(current_positions, initial_positions, q_vectors):
    """
    Compute the direction-averaged ISF for a given q magnitude.
    :param current_positions: Current positions of particles (Nx3 array).
    :param initial_positions: Initial positions of particles (Nx3 array).
    :param q_magnitude: Magnitude of the wave vector q.
    :param num_q_samples: Number of q vector samples for averaging.
    """
    N = len(current_positions)
    displacement = current_positions - initial_positions  # Nx3 array of displacements

    # Calculate the exponential term for each q vector and displacement
    exp_terms = np.exp(1j * np.dot(displacement, q_vectors.T))  # Nxnum_q_samples array

    # Average over all particles and q vectors
    average_isf = np.mean(exp_terms, axis=(0, 1))
    return average_isf
class ISF:
        def __init__(self,step_tot,log_check_points,coarse_grained_steps,gillespie,q_norm,num_q_samples,*args):
            self.q_vectors = q_norm * uniform_sphere_samples(num_q_samples)            
            self.gillespie = gillespie
            #self.isf_time = np.zeros((step_tot//check_steps,check_steps//coarse_grained_step),dtype=float)
            #self.isf_time = [np.zeros((end-start)//coarse_grained_steps, dtype=float) for start, end in zip([0]+log_check_points[:-1], log_check_points)]
            self.isf_time = [np.zeros((step_tot-initial_step)//coarse_grained_steps,dtype=float) for initial_step in [0]+log_check_points[:-1]]
            self.initial_positions = np.zeros((log_check_points.__len__(),gillespie.Nlinker,3),dtype=float)  # one initial position per isf
            self.next_index = 0 # index of the next isf to be computed
            self.next_time=np.zeros((log_check_points.__len__()),dtype=int)  # adjust the index for each isf
        def compute(self,time,move,i,t,*args):
            try:
                #self.isf_time[i][t] = np.linalg.norm(Compute_Average_ISF(self.gillespie.get_r(periodic=True),self.initial_positions,self.q_vectors))
                for n in range(self.next_index):
                    self.isf_time[n][self.next_time[n]] = np.linalg.norm(Compute_Average_ISF(self.gillespie.get_r(periodic=True),self.initial_positions[n],self.q_vectors))
                    self.next_time[n]+=1
            except IndexError:
                print(i)
                print(t)
                print(self.next_time)
                print(len(self.isf_time))
                print(self.isf_time[i].shape)
                #raise
        def start_check_step(self,*args):
            #self.initial_positions = copy.copy(self.gillespie.get_r(periodic=True))
            self.initial_positions[self.next_index] = copy.copy(self.gillespie.get_r(periodic=True))
            self.next_index+=1
            #print(self.next_time)
        def end_check_step(self,*args):
            return
        def close(self,output,*args):
            #output.put(('create_vlarray', ('/'+'S'+hex(self.gillespie.seed),'ISF' , self.isf_time)))
            output.put(('create_vlarray', ('/S'+hex(self.gillespie.seed), 'ISF', pt.Float64Atom(shape=()), self.isf_time)))