import numpy as np
from Reader import *

Fmin = lambda N,L,Eb : Eb*N - (L-N)*np.log(4*np.pi)# N is the number of linkers
#Fmax = lambda N,L,Eb : Eb*N - MinEnt(N,L)
Fmax = lambda N,L, E : -1.5*((N-1)*np.log(3*(N-1)/(2*np.pi*L)) -  1)-L*np.log(4*np.pi)+E*N



class Data_Treatement:
    def __init__(self,filename,data_type):
        self.data_type = data_type
        self.Read = CustomHDF5Reader(filename)
        self.Read.open()
        self.attributes = self.Read.get_header_attributes()
        #print(self.Read.list_measurements(self.Read.list_groups()[0]))
        if data_type not in self.Read.list_measurements(self.Read.list_groups()[0]):
            raise ValueError('data_type unknown')
        else:
            # data shape :
            # if the data is a scalar: data.shape = (Nsample,step_tot//coarse_steps)
            # if it is a function: data.shape = (Nsample,step_tot//check_steps,check_steps//coarse_stetps)
            self.data = np.array([self.Read.get_measurement_data(grp,data_type) for grp in self.Read.list_groups()])
        if self.data[0].shape[0] == self.Read.get_measurement_data(self.Read.list_groups()[0],"Check_Time").shape[0]:
            TimeType = "Check_Time"
        elif self.data[0].shape[0] == self.Read.get_measurement_data(self.Read.list_groups()[0],"Coarse_Time").shape[0]:
            TimeType = "Coarse_Time"
        else:
            print(self.data[0].shape)
            print(self.Read.get_measurement_data(self.Read.list_groups()[0],"Coarse_Time").shape)
            print(self.Read.get_measurement_data(self.Read.list_groups()[0],"Check_Time").shape)
            raise ValueError('No time with correct shape found')
        self.time = np.array([self.Read.get_measurement_data(grp,TimeType) for grp in self.Read.list_groups()])
        self.Nsample = len(self.Read.list_groups())
        self.Read.close()
    def average(self,num_bins=100,log_scale=False):
        if len(self.time.shape)==2: # which mean average a scalar:
            if self.data_type == 'cluster':
                self.variance,self.average_data = np.zeros((num_bins,3)),np.zeros((num_bins,3))
                for i in range(self.data.shape[-1]):
                    self.binned_time,self.average_data[:,i],self.variance[:,i] = average_scalar(self.time,self.data[:,:,i],num_bins,log_scale=log_scale)
            #self.binned_time,self.average_data,self.variance,self.distribution = average_scalar(self.time,self.data,num_bins)
            else:
                self.binned_time,self.average_data,self.variance = average_scalar(self.time,self.data,num_bins,log_scale=log_scale)
        elif len(self.time.shape)==3: # for MSD and ISF for instance
            self.binned_time,self.average_data,self.variance = np.zeros((self.data.shape[1],num_bins),dtype=float),np.zeros((self.data.shape[1],num_bins),dtype=float),np.zeros((self.data.shape[1],num_bins),dtype=float)
            for index in range(self.data.shape[1]):
                self.binned_time[index],self.average_data[index],self.variance[index] = average_scalar(self.time[:,index],self.data[:,index],num_bins=num_bins,log_scale=log_scale)
        else:
            raise IndexError('Invalid time shape')
    def rescale_energy(self):
            if self.data_type!='NRG':
                raise ValueError('wrong data type')            
            N = self.attributes['Nlinker']
            L = self.attributes['ell_tot']
            E = self.attributes['Energy']
            for n in range(self.Nsample):
                self.data[n] = ( self.data[n] - Fmin(N,L,E))/(Fmax(N,L,E)-Fmin(N,L,E))
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
#def average_scalar(X, Y, num_bins=100):
#    bins = np.linspace(X.min(), X.max(), num_bins + 1)
#    bin_centers = (bins[:-1] + bins[1:]) / 2
#    average_scalar = np.zeros(num_bins)
#    variance_scalar = np.zeros(num_bins) + np.nan  # Initialize with NaN
#    #distribution = []
#    
#    for i in range(num_bins):
#        scalar_values = []
#        for j in range(Y.shape[0]):
#            indices = np.digitize(X[j], bins) - 1
#            indices = np.clip(indices, 0, num_bins-1)
#            #if i==num_bins - 10:
#            #    distribution.append(Y[j][indices==i])
#            within_bin = Y[j][indices == i]
#            if within_bin.size > 0:
#                scalar_values.extend(within_bin)
#        if scalar_values:
#            average_scalar[i] = np.mean(scalar_values)            
#            if len(scalar_values) > 1:  # Ensure at least two data points for variance
#                variance_scalar[i] = np.std(scalar_values, ddof=1)
#            else:
#                variance_scalar[i] = 0  # Or set to 0, if that's more appropriate for your analysis
#            
#    return bin_centers, average_scalar, variance_scalar#, distribution

