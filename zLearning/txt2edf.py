import pandas as pd
import pyedflib
import numpy as np
import re

sampling_rate = 125
EEG_txt_file = 'EEG.txt'
EMG_txt_file = 'EMG.txt'
edf_file = 'edf_file.edf'

def txt2df(EEG_txt_file, EMG_txt_file):
    # transform txt file into df
    dict = {'EEG': [], 'EMG': []}

    with open(EEG_txt_file, 'r') as f:
        for line in f:
            line = float(re.sub('\n', '', line))
            dict['EEG'].append(line)

    with open(EMG_txt_file, 'r') as f:
        for line in f:
            line = float(re.sub('\n', '', line))
            dict['EMG'].append(line)

    df = pd.DataFrame(dict)

    print(df)
    return df

def df2edf(edf_file, sampling_rate):
    # transform df into edf
    channel_labels = df.columns.tolist()
    data = df.to_numpy().T
    signal_headers = pyedflib.highlevel.make_signal_headers(channel_labels)

    for header in signal_headers:
            header['sample_rate'] = sampling_rate
        
    pyedflib.highlevel.write_edf(edf_file, data, signal_headers)

df = txt2df(EEG_txt_file, EMG_txt_file)
df2edf(edf_file, sampling_rate)



#%%


import mne
file = "edf_file.edf"
data = mne.io.read_raw_edf(file)
raw_data = data.get_data()
# you can get the metadata included in the file and a list of all channels:
info = data.info
channels = data.ch_names

print(info)
print(channels)