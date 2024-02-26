import numpy as np
from Reader import *
class Data_Treatement:
    def __init__(self,filename,data_type):
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
        if self.data[0].shape == self.Read.get_measurement_data(self.Read.list_groups()[0],"Check_Time").shape:
            TimeType = "Check_Time"
        elif self.data[0].shape == self.Read.get_measurement_data(self.Read.list_groups()[0],"Coarse_Time").shape:
            TimeType = "Coarse_Time"
        else:
            print(self.data[0].shape)
            print(self.Read.get_measurement_data(self.Read.list_groups()[0],"Coarse_Time").shape)
            print(self.Read.get_measurement_data(self.Read.list_groups()[0],"Check_Time").shape)
            raise ValueError('No time with correct shape found')
        self.time = np.array([self.Read.get_measurement_data(grp,TimeType) for grp in self.Read.list_groups()])
        self.Nsample = len(self.Read.list_groups())
        self.Read.close()
    def average(self,num_bins=100):
        if len(self.time.shape)==2: # which mean average a scalar:
            self.binned_time,self.average_data,self.variance = average_scalar(self.time,self.data,num_bins)
        elif len(self.time.shape)==3: # for MSD and ISF for instance
            self.binned_time,self.average_data,self.variance = np.zeros((self.data.shape[1],num_bins),dtype=float),np.zeros((self.data.shape[1],num_bins),dtype=float),np.zeros((self.data.shape[1],num_bins),dtype=float)
            for index in range(self.data.shape[1]):
                self.binned_time[index],self.average_data[index],self.variance[index] = average_scalar(self.time[:,index],self.data[:,index])
        else:
            raise IndexError('Invalid time shape')
    
def average_scalar(X, Y, num_bins=100):
    bins = np.linspace(X.min(), X.max(), num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    average_scalar = np.zeros(num_bins)
    variance_scalar = np.zeros(num_bins) + np.nan  # Initialize with NaN
    
    for i in range(num_bins):
        scalar_values = []
        for j in range(Y.shape[0]):
            indices = np.digitize(X[j], bins) - 1
            indices = np.clip(indices, 0, num_bins-1)
            within_bin = Y[j][indices == i]
            if within_bin.size > 0:
                scalar_values.extend(within_bin)
        if scalar_values:
            average_scalar[i] = np.mean(scalar_values)
            if len(scalar_values) > 1:  # Ensure at least two data points for variance
                variance_scalar[i] = np.var(scalar_values, ddof=1)
            else:
                variance_scalar[i] = 0  # Or set to 0, if that's more appropriate for your analysis
            
    return bin_centers, average_scalar, variance_scalar