#%%
'''
Find out which kind of FF file it is and create easy csv file from it
'''

import h5py
import numpy as np
import pandas as pd
import os
path = 'C:\\Users\\landgrafn\\Desktop\\ff\\'
rec_type = None


def get_files(path, common_name):

    # get files
    files = [os.path.join(path, file) for file in os.listdir(path) 
             if os.path.isfile(os.path.join(path, file)) and common_name in file]    
    
    print(f'{len(files)} files found')
    for file in files:
        print(f'{file}')
    print('\n')

    return files

def get_rec_type(file):
    # returns the rec_type as a string

    path = 'DataAcquisition/NC500/Signals/Series0001/'
    paths = {'NormFilter': path + 'AnalogIn/AIN01',
             'LockIn'    : path + 'LockInAOUT01/AIN01'}

    with h5py.File(file, 'r') as f:
        for rec_type, hdf5_path in paths.items():

            try:
                _ = np.array(f[hdf5_path])
                return rec_type
            
            except KeyError:
                continue

    return ''

def do_NormFilter(file):

    with h5py.File(file, 'r') as f:
        path = 'DataAcquisition/NC500/Signals/Series0001/'

        time_sec    = np.array(f[path + 'AnalogIn/Time'])
        sig         = np.array(f[path + 'AnalogIn/AIN01'])

        # figure out which output channels are available
        output_channels, i = {'Isos': 'AnalogOut/AOUT01', 'Fluo': 'AnalogOut/AOUT02'}, 0
        for channel_name, channel_descrip in output_channels.items():
            try:
                output = np.array(f[path + channel_descrip])
                name = channel_name
                i += 1
            except KeyError:
                continue

        output = np.round(output, 1)
        duration = round(max(time_sec), 1)
        sampling_rate = round(len(sig) / duration)

        
        # create quick overview
        if len(time_sec) == len(sig) == len(output):
            datapoints = f'{len(time_sec)}=Time=Signal=Output'  
        else:
            datapoints = f'{len(time_sec)} (Time), {len(sig)} (Signal), {len(output)} (Output)'

        print(  f'NormFilter: {name} ({i} channel)\n'
                f'Output: {[min(output), np.mean(output), max(output)]} V\n'
                f'Datapoints: {datapoints}\nDuration: {duration}s\nSampling_rate: {sampling_rate}Hz\n')

    # create data as .csv
    data = pd.DataFrame({'Time': time_sec, 'Signal': sig, 'Output': output})
    new_file = file.replace('.doric', '')
    title = f'{new_file}_normFilt_{name}({np.mean(output)}-{max(output)}V)'
    data.to_csv(f'{title}.csv', index=False)

def do_LockIn(file):

    with h5py.File(file, 'r') as f:
        path = 'DataAcquisition/NC500/Signals/Series0001/'

        time_sec    = np.array(f[path + 'LockInAOUT01/Time'])
        isos        = np.array(f[path + 'LockInAOUT01/AIN01'])
        fluo        = np.array(f[path + 'LockInAOUT02/AIN01'])
        output_isos = np.array(f[path + 'AnalogOut/AOUT01'])
        output_fluo = np.array(f[path + 'AnalogOut/AOUT02'])

        output_isos_min, output_isos_mean, output_isos_max = round(min(output_isos), 1), round(np.mean(output_isos), 1), round(max(output_isos), 1)
        output_fluo_min, output_fluo_mean, output_fluo_max = round(min(output_fluo), 1), round(np.mean(output_fluo), 1), round(max(output_fluo), 1)
        duration = max(time_sec)
        sampling_rate = round(len(isos) / duration, 2)

        print(  f'LockIn\n'
                f'Output_Isos: {[output_isos_min, output_isos_mean, output_isos_max]}V, {len(output_isos)}points\n'
                f'Output_Fluo: {[output_fluo_min, output_fluo_mean, output_fluo_max]}V, {len(output_fluo)}points\n'
                f'Datapoints: {len(time_sec)} (Time), {len(isos)} (Isos), {len(fluo)} (Fluo)\n'
                f'Duration: {duration}s\n'
                f'Sampling_rate: {sampling_rate}Hz\n')
        
    data = pd.DataFrame({'Time': time_sec, 'Isos': isos, 'Fluo': fluo})
    new_file = file.replace('.doric', '')
    title = f'{new_file}_LockIn_Isos({output_isos_mean}-{output_isos_max}V).Fluo({output_fluo_mean}-{output_fluo_max}V)'
    data.to_csv(f'{title}.csv', index=False)

def main():
    files = get_files(path, '.doric')
    for file in files:

        print(f'\n{file}')
        rec_type = get_rec_type(file)

        if rec_type == 'NormFilter':
            do_NormFilter(file)
        elif rec_type == 'LockIn':
            do_LockIn(file)

main()
