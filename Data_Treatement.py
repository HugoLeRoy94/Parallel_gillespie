import numpy as np
from Reader import *

Fmin = lambda N,L,Eb : Eb*N - (L-N)*np.log(4*np.pi)#-N*np.log(7)# N is the number of linkers
#Fmax = lambda N,L,Eb : Eb*N - MinEnt(N,L)
Fmax = lambda N,L, E : -1.5*((N-1)*np.log(3*(N-1)/(2*np.pi*L)) -  1)-L*np.log(4*np.pi)+E*N #- 3/2*N*np.log(L/N)

#def interpolate_empty_bins(data):
#    for i in range(len(data)):
#        if data[i] == 0:  # Assuming 0 indicates an empty bin
#            non_empty_neighbors = []
#            if i > 0 and data[i-1] != 0:
#                non_empty_neighbors.append(data[i-1])
#            if i < len(data) - 1 and data[i+1] != 0:
#                non_empty_neighbors.append(data[i+1])
#            
#            if non_empty_neighbors:
#                data[i] = sum(non_empty_neighbors) / len(non_empty_neighbors)
#    return data
def interpolate_empty_bins(data):
    # Identify indices of non-empty bins
    non_empty_indices = np.nonzero(data)[0]
    if len(non_empty_indices) == 0:
        # Handle case where data might be all zeros
        return data
    # Interpolate for empty bins by using np.interp
    # For np.interp, positions of the non-empty bins are the 'xp' and their values are 'fp'
    interpolated_data = np.interp(np.arange(len(data)), non_empty_indices, data[non_empty_indices])
    return interpolated_data

class Data_Treatement:
    def __init__(self,filename,data_type):
        self.data_type = data_type
        self.Read = CustomHDF5Reader(filename)
        self.Read.open()
        self.attributes = self.Read.get_header_attributes()
        #print(self.Read.list_measurements(self.Read.list_groups()[0]))
        if data_type not in self.Read.list_measurements(self.Read.list_groups()[0]):
            raise ValueError('data_type unknown')
        self.data = np.array([self.Read.get_measurement_data(grp,data_type) for grp in self.Read.list_groups()],dtype=object)
        if len(self.data[0]) == len(self.Read.get_measurement_data(self.Read.list_groups()[0],"Check_Time")):
            TimeType = "Check_Time"
        elif len(self.data[0]) == len(self.Read.get_measurement_data(self.Read.list_groups()[0],"Coarse_Time")):
            TimeType = "Coarse_Time"
        else:
            print(self.data[0].shape)
            print(self.Read.get_measurement_data(self.Read.list_groups()[0],"Coarse_Time").shape)
            print(self.Read.get_measurement_data(self.Read.list_groups()[0],"Check_Time").shape)
            raise ValueError('No time with correct shape found')
        self.time = np.array([self.Read.get_measurement_data(grp,TimeType) for grp in self.Read.list_groups()],dtype=object)
        self.Nsample = len(self.Read.list_groups())
        self.Read.close()
    def average(self,num_bins=100,log_scale=False):
        if self.data_type == 'cluster':
            self.variance,self.average_data = np.zeros((num_bins,3)),np.zeros((num_bins,3))
            for i in range(3):
                self.binned_time,self.average_data[:,i],self.variance[:,i] = average_scalar(self.time,self.data[:,:,i],num_bins,log_scale=log_scale)
                self.average_data[:,i] = interpolate_empty_bins(self.average_data[:,i])
            #self.binned_time,self.average_data,self.variance,self.distribution = average_scalar(self.time,self.data,num_bins)
        elif self.data_type in {'NRG','MSD_tot','Coarse_Time'}:
                self.binned_time,self.average_data,self.variance = average_scalar(self.time,self.data,num_bins,log_scale=log_scale)
                self.average_data = interpolate_empty_bins(self.average_data)
        elif self.data_type in {'PCF','PCF_L'}:
            self.binned_time,self.average_data,self.variance = np.zeros((self.data.shape[1],self.data.shape[2]),dtype=float),np.zeros((self.data.shape[1],self.data.shape[2]),dtype=float),np.zeros((self.data.shape[1],self.data.shape[2]),dtype=float)
            for index in range(self.data.shape[1]): # loop over the different timesteps
                self.binned_time[index],self.average_data[index],self.variance[index] = average_scalar(self.data[:,index,:,0],self.data[:,index,:,1],num_bins=self.data.shape[2],log_scale=log_scale)
                self.average_data[index] = interpolate_empty_bins(self.average_data[index])
        elif self.data_type in {'MSD','ISF'}:
            self.binned_time,self.average_data,self.variance = np.zeros((len(self.data[0]),num_bins),dtype=float),np.zeros((len(self.data[0]),num_bins),dtype=float),np.zeros((len(self.data[0]),num_bins),dtype=float)
            for index in range(len(self.data[0])):
                times = np.array([self.time[i][index] for i in range(self.time.shape[0])])
                datas = np.array([self.data[i][index] for i in range(self.data.shape[0])])
                self.binned_time[index],self.average_data[index],self.variance[index] = average_scalar(times,datas,num_bins=num_bins,log_scale=log_scale)
                self.average_data[index] = interpolate_empty_bins(self.average_data[index])
        else:
            raise IndexError('data-type does not correspond to any known average')
    def rescale_energy(self):
            if self.data_type!='NRG':
                raise ValueError('wrong data type')            
            N = self.attributes['Nlinker']
            L = self.attributes['ell_tot']
            E = self.attributes['Energy']
            for n in range(self.Nsample):
                self.data[n] = ( self.data[n] - Fmin(N,L,E))/(Fmax(N,L,E)-Fmin(N,L,E))
#def average_scalar(X, Y, num_bins=100, log_scale=False, min_bin_size=1):
#    if log_scale:
#        # Ensure X contains only positive values for logarithmic scale
#        X = np.maximum(X, min_bin_size)
#        min_power = np.log10(X.min())
#        max_power = np.log10(X.max())
#        bins = np.logspace(min_power, max_power, num_bins + 1)
#        bin_centers = (bins[:-1] * bins[1:]) ** 0.5  # Geometric mean for bin centers
#    else:
#        bins = np.linspace(X.min(), X.max(), num_bins + 1)
#        bin_centers = (bins[:-1] + bins[1:]) / 2
#    
#    average_scalar = np.zeros(num_bins)
#    variance_scalar = np.zeros(num_bins) + np.nan  # Initialize with NaN
#
#    for i in range(num_bins):
#        scalar_values = []
#        for j in range(Y.shape[0]):
#            indices = np.digitize(X[j], bins) - 1  # Find indices of bins into which each X[j] falls
#            indices = np.clip(indices, 0, num_bins-1)  # Clip indices to valid range
#            within_bin = Y[j][indices == i]  # Select Y values that fall into the current bin
#            if within_bin.size > 0:
#                scalar_values.extend(within_bin)  # Collect all Y values in the current bin across all curves
#        if scalar_values:
#            average_scalar[i] = np.mean(scalar_values)  # Compute average of scalar values in the bin
#            if len(scalar_values) > 1:  # Compute variance if there are at least two values
#                variance_scalar[i] = np.std(scalar_values, ddof=1)  # Sample standard deviation for variance
#            else:
#                variance_scalar[i] = 0  # Set variance to 0 if only one value in the bin
#
#    return bin_centers, average_scalar, variance_scalar
def average_scalar(X, Y, num_bins=100, log_scale=False, min_bin_size=1):
    # Ensure X is at least 2D
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    
    # Adjust X for logarithmic scale if requested
    if log_scale:
        X = np.maximum(X, min_bin_size)
        bins = np.logspace(np.log10(X.min()), np.log10(X.max()), num_bins + 1)
    else:
        bins = np.linspace(X.min(), X.max(), num_bins + 1)
    
    bin_centers = (bins[:-1] + bins[1:]) / 2
    if log_scale:
        bin_centers = np.sqrt(bins[:-1] * bins[1:])
    
    # Pre-calculate bin indices for all X values
    bin_indices = np.digitize(X.ravel(), bins) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins-1)

    # Initialize arrays for average and variance
    average_scalar = np.zeros(num_bins)
    variance_scalar = np.zeros(num_bins) + np.nan

    for i in range(num_bins):
        # Boolean array indicating whether each element falls into the current bin
        in_bin = bin_indices == i
        if np.any(in_bin):
            # Compute average directly
            average_scalar[i] = np.mean(Y.ravel()[in_bin])
            # Compute variance, using ddof=1 for sample standard deviation
            if np.sum(in_bin) > 1:
                variance_scalar[i] = np.var(Y.ravel()[in_bin], ddof=1)
            else:
                variance_scalar[i] = 0

    return bin_centers, average_scalar, variance_scalar