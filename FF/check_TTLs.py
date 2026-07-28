#%%
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, optimize, stats
from pathlib import Path
import os

file = r"C:\Users\landgrafn\Downloads\timed.doric"



def get_data(file):

    with h5py.File(file, 'r') as f:
        path = 'DataAcquisition/NC500/Signals/Series0001/'

    
        digital_io  = np.array(f[path + 'DigitalIO/DIO01'])
        digital_time= np.array(f[path + 'DigitalIO/Time'])


        

        return digital_io, digital_time
    

print(file)
digital_io, digital_time = get_data(file)



df = pd.DataFrame(digital_io)
print(df)

# Get the second column (signal)
signal = df[0]

# Compute difference between consecutive values
# A transition from 0.0 to 1.0 will result in a difference of +1.0
transitions = (signal.diff() == 1.0).sum()

print(f"Number of 0.0 → 1.0 transitions: {int(transitions)}")




