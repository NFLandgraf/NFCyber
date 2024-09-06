import h5py # Make sure to install the library
import numpy as np
import pandas as pd

# Add filepath to your file
dataframe_1 = 'C:\\Users\\landgrafn\\NFCyber\\FFAnalysis'

# Read datasets from the hdf5 file
with h5py.File(dataframe_1, 'r') as f:
	# Confirm dataset paths in the HDF5 Viewer
	path = '/DataAcquisition/EPConsole/Signals/Series0001/'
	time_     = np.array(f[path+'AIN01xAOUT01-LockIn/Time'])
	dio1      = np.array(f[path+'/DigitalIO/DIO1'])
	ain1aout1 = np.array(f[path+'/AIN01xAOUT01-LockIn/Values'])
	dio2      = np.array(f[path+'/DigitalIO/DIO2'])
	ain1aout2 = np.array(f[path+'/AIN01xAOUT02-LockIn/Values'])

# Create data frame
df = pd.DataFrame({
	'Time(s)' : time_,
	'Tone'    : dio1,
	'isobestic': ain1aout1,
	'Lick'     : dio2,
	'Photosignal': ain1aout2
	})

print(df.head())