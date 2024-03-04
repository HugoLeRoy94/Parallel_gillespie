import numpy as np
from Reader import *

Fmin = lambda N,L,Eb : Eb*N - (L-N)*np.log(4*np.pi)#-N*np.log(7)# N is the number of linkers
#Fmax = lambda N,L,Eb : Eb*N - MinEnt(N,L)
Fmax = lambda N,L, E : -1.5*((N-1)*np.log(3*(N-1)/(2*np.pi*L)) -  1)-L*np.log(4*np.pi)+E*N #- 3/2*N*np.log(L/N)



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
            #self.binned_time,self.average_data,self.variance,self.distribution = average_scalar(self.time,self.data,num_bins)
        elif self.data_type in {'NRG','MSD_tot','Coarse_Time'}:
                self.binned_time,self.average_data,self.variance = average_scalar(self.time,self.data,num_bins,log_scale=log_scale)
        elif self.data_type in {'PCF','PCF_L'}:
            self.binned_time,self.average_data,self.variance = np.zeros((self.data.shape[1],self.data.shape[2]),dtype=float),np.zeros((self.data.shape[1],self.data.shape[2]),dtype=float),np.zeros((self.data.shape[1],self.data.shape[2]),dtype=float)
            for index in range(self.data.shape[1]): # loop over the different timesteps
                self.binned_time[index],self.average_data[index],self.variance[index] = average_scalar(self.data[:,index,:,0],self.data[:,index,:,1],num_bins=self.data.shape[2],log_scale=log_scale)
        elif self.data_type in {'MSD','ISF'}:
            self.binned_time,self.average_data,self.variance = np.zeros((len(self.data[0]),num_bins),dtype=float),np.zeros((len(self.data[0]),num_bins),dtype=float),np.zeros((len(self.data[0]),num_bins),dtype=float)
            for index in range(len(self.data[0])):
                times = np.array([self.time[i][index] for i in range(self.time.shape[0])])
                datas = np.array([self.data[i][index] for i in range(self.data.shape[0])])
                self.binned_time[index],self.average_data[index],self.variance[index] = average_scalar(times,datas,num_bins=num_bins,log_scale=log_scale)
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
#def average_scalar_list(X, Y, num_bins=100, log_scale=False, min_bin_size=1):
#    if log_scale:
#        # Flatten X to get the global min and max for logscale bins
#        X_flat = np.concatenate(X)
#        X_flat = np.maximum(X_flat, min_bin_size)  # Ensure positive values for log scale
#        min_power = np.log10(X_flat.min())
#        max_power = np.log10(X_flat.max())
#        bins = np.logspace(min_power, max_power, num_bins + 1)
#    else:
#        # Flatten X for linear bins
#        X_flat = np.concatenate(X)
#        bins = np.linspace(X_flat.min(), X_flat.max(), num_bins + 1)
#    bin_centers = (bins[:-1] + bins[1:]) / 2  # Adjust for linear or log scale
#
#    average_scalar = np.zeros(num_bins)
#    variance_scalar = np.zeros(num_bins)
#
#    bin_counts = np.zeros(num_bins)
#    sum_scalar = np.zeros(num_bins)
#    sum_square_scalar = np.zeros(num_bins)
#
#    for x, y in zip(X, Y):  # X and Y are lists of arrays
#        indices = np.digitize(x, bins) - 1  # Find bin for each x
#        indices = np.clip(indices, 0, num_bins - 1)  # Ensure valid index range
#
#        for i in range(num_bins):
#            in_bin = y[indices == i]  # Select y values in current bin
#            if in_bin.size > 0:
#                bin_counts[i] += in_bin.size
#                sum_scalar[i] += in_bin.sum()
#                sum_square_scalar[i] += np.sum(in_bin**2)
#
#    # Compute average and variance where bin_counts > 0 to avoid division by zero
#    valid_bins = bin_counts > 0
#    average_scalar[valid_bins] = sum_scalar[valid_bins] / bin_counts[valid_bins]
#    # Compute variance using the formula var = E[X^2] - (E[X])^2
#    variance_scalar[valid_bins] = sum_square_scalar[valid_bins] / bin_counts[valid_bins] - average_scalar[valid_bins]**2
#
#    return bin_centers, average_scalar, variance_scalar
def average_scalar(X, Y, num_bins=100, log_scale=False, min_bin_size=1):
    if log_scale:
        # Ensure X contains only positive values for logarithmic scale
        X = np.maximum(X, min_bin_size)
        min_power = np.log10(X.min())
        max_power = np.log10(X.max())
        bins = np.logspace(min_power, max_power, num_bins + 1)
        bin_centers = (bins[:-1] * bins[1:]) ** 0.5  # Geometric mean for bin centers
    else:
        bins = np.linspace(X.min(), X.max(), num_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
    
    average_scalar = np.zeros(num_bins)
    variance_scalar = np.zeros(num_bins) + np.nan  # Initialize with NaN

    for i in range(num_bins):
        scalar_values = []
        for j in range(Y.shape[0]):
            indices = np.digitize(X[j], bins) - 1  # Find indices of bins into which each X[j] falls
            indices = np.clip(indices, 0, num_bins-1)  # Clip indices to valid range
            within_bin = Y[j][indices == i]  # Select Y values that fall into the current bin
            if within_bin.size > 0:
                scalar_values.extend(within_bin)  # Collect all Y values in the current bin across all curves
        if scalar_values:
            average_scalar[i] = np.mean(scalar_values)  # Compute average of scalar values in the bin
            if len(scalar_values) > 1:  # Compute variance if there are at least two values
                variance_scalar[i] = np.std(scalar_values, ddof=1)  # Sample standard deviation for variance
            else:
                variance_scalar[i] = 0  # Set variance to 0 if only one value in the bin

    return bin_centers, average_scalar, variance_scalar
