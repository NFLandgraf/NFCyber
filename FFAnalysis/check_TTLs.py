#%%
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, optimize, stats
from pathlib import Path
import os

file = r"C:\Users\landgrafn\Desktop\checko\FF_Test_10Hz.doric"



def get_data(file):

    with h5py.File(file, 'r') as f:
        path = 'DataAcquisition/NC500/Signals/Series0001/'

    
        digital_io  = np.array(f[path + 'DigitalIO/DIO05'])
        digital_time= np.array(f[path + 'DigitalIO/Time'])


        

        return digital_io, digital_time
    

print(file)
digital_io, digital_time = get_data(file)

#%%
print(digital_io)



df = pd.DataFrame(digital_io)
df.to_csv('10Hz.csv')
print(df)



#%%

import pandas as pd

# Replace with your actual CSV file path
csv_file = r"C:\Users\landgrafn\NFCyber\FFAnalysis\30Hz.csv"

# Load the CSV (no header)
df = pd.read_csv(csv_file, header=None)

# Get the second column (signal)
signal = df[1]

# Compute difference between consecutive values
# A transition from 0.0 to 1.0 will result in a difference of +1.0
transitions = (signal.diff() == 1.0).sum()

print(f"Number of 0.0 → 1.0 transitions: {int(transitions)}")




