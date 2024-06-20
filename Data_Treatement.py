import numpy as np
from Reader import *

Fmin = lambda N,L,Eb : Eb*N - (L-N)*np.log(4*np.pi)#-N*np.log(7)# N is the number of linkers
#Fmax = lambda N,L,Eb : Eb*N - MinEnt(N,L)
Fmax = lambda N,L, E : -1.5*((N-1)*np.log(3*(N-1)/(2*np.pi*L)) -  1)-L*np.log(4*np.pi)+E*N #- 3/2*N*np.log(L/N)

from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
norm = ImageNormalize(vmin=0., vmax=100, stretch=LogStretch())
import mpl_scatter_density # adds projection='scatter_density'
from matplotlib.colors import LinearSegmentedColormap

# "Viridis-like" colormap with white background
white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
    (0, '#ffffff'),
    (1e-20, '#440053'),
    (0.2, '#404388'),
    (0.4, '#2a788e'),
    (0.6, '#21a784'),
    (0.8, '#78d151'),
    (1, '#fde624'),
], N=256)

def sliding_average(X, Y, window_size=5):
    """
    Apply a sliding average (moving average) to the curve defined by (X, Y).
    
    Parameters:
    - X: numpy array of x-values.
    - Y: numpy array of y-values, must be the same length as X.
    - window_size: size of the sliding window (number of points to average).
    
    Returns:
    - X_smooth: X values corresponding to the center of each sliding window.
    - Y_smooth: Smoothed Y values.
    """
    half_window = window_size // 2
    
    # Initialize smoothed Y array
    Y_smooth = np.convolve(Y, np.ones(window_size)/window_size, mode='valid')
    
    # Adjust X to match the size of the smoothed Y array
    # This centers the window on the point being averaged
    start_index = (window_size - 1) // 2
    end_index = start_index + len(Y_smooth)
    X_smooth = X[start_index:end_index]
    
    return X_smooth, Y_smooth

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
def scatter_density(fig, x, y,nrows=1,ncols=1,pos=1,dpi=75,norm=None,colorbar=True,cmap=white_viridis,ax=None,xlim=None,ylim=None,resample=False,LogStretch = False,vmin =None,vmax=None):
    if ax is None:
        ax = fig.add_subplot(nrows, ncols, pos, projection='scatter_density')
    if xlim is not None:
        x_mask = (x >= xlim[0]) & (x <= xlim[1])
        x = x[x_mask]
        y = y[x_mask]
        ax.set_xlim(xlim[0],xlim[1])
    if ylim is not None:
        y_mask = (y >= ylim[0]) & (y <= ylim[1])
        y = y[y_mask]
        x = x[y_mask]
        ax.set_ylim(ylim[0],ylim[1])
    if resample :
        weights = 1 / x
        # Step 2: Resample your data according to weights
        # This is a simplified approach; for large datasets, consider a more efficient resampling method.
        resampled_indices = np.random.choice(a=range(x.size), size=x.size, replace=True, p=weights/weights.sum())
        x = x[resampled_indices]
        y = y[resampled_indices]
    
    if norm is None or norm is False:
        density = ax.scatter_density(x, y, cmap=cmap,dpi=dpi)
    else:
        if LogStretch:
            norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LogStretch())
        else:
            norm = ImageNormalize(vmin=vmin, vmax=vmax)
        density = ax.scatter_density(x, y, cmap=cmap,dpi=dpi,norm=norm)
    if colorbar:
        fig.colorbar(density, label='Number of points per pixel')
    return ax
class Data_Treatement:
    def __init__(self,filename,data_type):
        self.data_type = data_type
        self.Read = CustomHDF5Reader(filename)
        self.Read.open()
        self.attributes = self.Read.get_header_attributes()
        #print(self.Read.list_measurements(self.Read.list_groups()[0]))
        if data_type not in self.Read.list_measurements(self.Read.list_groups()[0]):            
            raise ValueError('data_type unknown')
        self.data = np.array([self.Read.get_measurement_data(grp,data_type) for grp in self.Read.list_groups() if self.Read.get_measurement_data(grp,data_type) is not None],dtype=object)
        if len(self.data[0]) == len(self.Read.get_measurement_data(self.Read.list_groups()[0],"Check_Time")):
            TimeType = "Check_Time"
        elif len(self.data[0]) == len(self.Read.get_measurement_data(self.Read.list_groups()[0],"Coarse_Time")):
            TimeType = "Coarse_Time"
        else:
            print(self.data[0].shape)
            print(self.Read.get_measurement_data(self.Read.list_groups()[0],"Coarse_Time").shape)
            print(self.Read.get_measurement_data(self.Read.list_groups()[0],"Check_Time").shape)
            raise ValueError('No time with correct shape found')
        if data_type !='ISF':
            self.time = np.array([self.Read.get_measurement_data(grp,TimeType) for grp in self.Read.list_groups() if self.Read.get_measurement_data(grp,TimeType) is not None],dtype=object)
        else:
            coarse_time = np.array([self.Read.get_measurement_data(grp,"Coarse_Time") for grp in self.Read.list_groups() if self.Read.get_measurement_data(grp,"Coarse_Time") is not None],dtype=object)
            self.time = [[t_sys[-isf.__len__():] - t_sys[-isf.__len__()] for isf in system]for  t_sys,system in zip(coarse_time,self.data)]
        self.Nsample = len(self.Read.list_groups())
        self.Read.close()
    def average(self,num_bins=100,log_scale=False,min_bin_val = None):
        #if np.any(self.data==None):
        if np.any([x is None for x in self.data]):
            print('Nones founds in the array, certainly due to defective seeds, we remove it before averaging')
            self.data = self.data[self.data!=None]
        if self.data_type == 'cluster':
            self.variance,self.average_data = np.zeros((num_bins,3)),np.zeros((num_bins,3))
            for i in range(3):
                self.binned_time,self.average_data[:,i],self.variance[:,i] = average_scalar(self.time,self.data[:,:,i],num_bins,log_scale=log_scale,min_bin_val=min_bin_val)
                self.average_data[:,i] = interpolate_empty_bins(self.average_data[:,i])
            #self.binned_time,self.average_data,self.variance,self.distribution = average_scalar(self.time,self.data,num_bins)
        elif self.data_type in {'NRG','MSD_tot','Coarse_Time','Entropy'}:
                self.binned_time,self.average_data,self.variance = average_scalar(self.time,self.data,num_bins,log_scale=log_scale,min_bin_val=min_bin_val)
                self.average_data = interpolate_empty_bins(self.average_data)
        elif self.data_type in {'PCF','PCF_L'}:
            self.binned_time,self.average_data,self.variance = np.zeros((self.data.shape[1],self.data.shape[2]),dtype=float),np.zeros((self.data.shape[1],self.data.shape[2]),dtype=float),np.zeros((self.data.shape[1],self.data.shape[2]),dtype=float)
            for index in range(self.data.shape[1]): # loop over the different timesteps
                self.binned_time[index],self.average_data[index],self.variance[index] = average_scalar(self.data[:,index,:,0],self.data[:,index,:,1],num_bins=self.data.shape[2],log_scale=log_scale,min_bin_val=min_bin_val)
                self.average_data[index] = interpolate_empty_bins(self.average_data[index])
        elif self.data_type in {'MSD','ISF'}:
            self.binned_time,self.average_data,self.variance = np.zeros((len(self.data[0]),num_bins),dtype=float),np.zeros((len(self.data[0]),num_bins),dtype=float),np.zeros((len(self.data[0]),num_bins),dtype=float)
            for index in range(len(self.data[0])):
                times = np.array([self.time[i][index] for i in range(self.time.__len__())])
                datas = np.array([self.data[i][index] for i in range(self.data.shape[0])])
                self.binned_time[index],self.average_data[index],self.variance[index] = average_scalar(times,datas,num_bins=num_bins,log_scale=log_scale,min_bin_val=min_bin_val)
                self.average_data[index] = interpolate_empty_bins(self.average_data[index])
        else:
            raise IndexError('data-type does not correspond to any known average')
    def curate_data(self,imin,imax,window_size=10):
        if not hasattr(self,'binned_time') or not hasattr(self,'average_data') :
            self.average()
        X,Y = self.binned_time,self.average_data
        X,Y = X[imin:imax],Y[imin:imax]
        if window_size:
            X,Y = sliding_average(X,Y,window_size=window_size)
        Y = (Y - np.mean(Y[-len(Y)//10:]))/(np.mean(Y[:10])- np.mean(Y[-len(Y)//10:]))
        return X,Y
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
#def average_scalar(X, Y, num_bins=100, log_scale=False, min_bin_size=1):
#    # Ensure X is at least 2D
#    X = np.atleast_2d(X)
#    Y = np.atleast_2d(Y)
#    
#    # Adjust X for logarithmic scale if requested
#    if log_scale:
#        X = np.maximum(X, min_bin_size)
#        bins = np.logspace(np.log10(X.min()), np.log10(X.max()), num_bins + 1)
#    else:
#        bins = np.linspace(X.min(), X.max(), num_bins + 1)
#    
#    bin_centers = (bins[:-1] + bins[1:]) / 2
#    if log_scale:
#        bin_centers = np.sqrt(bins[:-1] * bins[1:])
#    
#    # Pre-calculate bin indices for all X values
#    bin_indices = np.digitize(X.ravel(), bins) - 1
#    bin_indices = np.clip(bin_indices, 0, num_bins-1)
#
#    # Initialize arrays for average and variance
#    average_scalar = np.zeros(num_bins)
#    variance_scalar = np.zeros(num_bins) + np.nan
#
#    for i in range(num_bins):
#        # Boolean array indicating whether each element falls into the current bin
#        in_bin = bin_indices == i
#        if np.any(in_bin):
#            # Compute average directly
#            average_scalar[i] = np.mean(Y.ravel()[in_bin])
#            # Compute variance, using ddof=1 for sample standard deviation
#            if np.sum(in_bin) > 1:
#                variance_scalar[i] = np.var(Y.ravel()[in_bin], ddof=1)
#            else:
#                variance_scalar[i] = 0
#
#    return bin_centers, average_scalar, variance_scalar
def average_scalar(X, Y, num_bins=100, log_scale=False, min_bin_val=None):
    # Ensure X is at least 2D
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    
    # Adjust X for logarithmic scale if requested
    if log_scale:
        if min_bin_val:
            X = np.maximum (X, min_bin_val)
        bins = np.logspace(np.log10(X.min()), np.log10(X.max()), num_bins + 1)
    else:
        bins = np.linspace(X.min(), X.max(), num_bins + 1)
    #print(bins)
    #print(X.min())
    #print(np.log10(X.min()))

    bin_centers = (bins[:-1] + bins[1:]) / 2
    if log_scale:
        bin_centers = np.sqrt(bins[:-1] * bins[1:])
    
    # Initialize arrays for the weighted average and variance
    weighted_average = np.zeros(num_bins)
    count = np.zeros(num_bins)
    # Loop over each curve
    for curve_x, curve_y in zip(X, Y):
        # Calculate dx for this curve
        dx = np.diff(np.append(0,curve_x), prepend=0)[1:]
        
        # Pre-calculate bin indices for all x values of this curve
        bin_indices = np.digitize(curve_x, bins) - 1
        bin_indices = np.clip(bin_indices, 0, num_bins-1)
        
        # Add the weighted average to the respective bin
        for i in range(num_bins):
            in_bin = bin_indices == i
            if np.any(in_bin):
                # Update the weighted average and weight sums
                weighted_average[i] += np.sum(curve_y[in_bin]*dx[in_bin])/np.sum(dx[in_bin])
                count[i]+=1
    weighted_average /=count
                
    
    # Variance calculation would need adjustment for this approach
    # It's complex since we're working with weighted averages per curve, not individual samples
    
    return bin_centers, weighted_average,0