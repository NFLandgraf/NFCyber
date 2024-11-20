#%%
'''
Find out which kind of FF file it is and create easy csv file from it, no preprocessing!
'''

import h5py
import numpy as np
import pandas as pd
import os
path = 'C:\\Users\\landgrafn\\Desktop\\2024-11-19_FF-Weilin-3m_YMaze\\'
common_name = '.doric'
rec_type = None


def get_files(path):

    # get files
    files = [os.path.join(path, file) for file in os.listdir(path) 
             if os.path.isfile(os.path.join(path, file)) and common_name in file]    
    
    print(f'{len(files)} files found')
    for file in files:
        print(f'{file}')
    print('\n')

    return files

def event_list(df, channel_name, value_threshold=0.9):
    # returns a list with the onset (and offset) timepoints, where certain channel is 'high' 
    
    # iterate through rows and save on- and offsets
    high_times, curr_high_time = [], []
    value_low = True

    for idx, value in df[channel_name].items():
        if value_low and value >= value_threshold:
            curr_high_time.append(idx)
            value_low = False
        elif not value_low and value <= value_threshold:
            curr_high_time.append(idx)
            high_times.append(curr_high_time)
            curr_high_time = []
            value_low = True

    return high_times

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

def do_LockIn_IO(file):

    with h5py.File(file, 'r') as f:
        path = 'DataAcquisition/NC500/Signals/Series0001/'

        # collect raw data from h5 file
        time_sec    = np.array(f[path + 'LockInAOUT01/Time'])
        isos        = np.array(f[path + 'LockInAOUT01/AIN01'])
        fluo        = np.array(f[path + 'LockInAOUT02/AIN01'])
        output_isos = np.array(f[path + 'AnalogOut/AOUT01'])
        output_fluo = np.array(f[path + 'AnalogOut/AOUT02'])
        digi_io     = np.array(f[path + 'DigitalIO/DIO02'])
        digi_time   = np.array(f[path + 'DigitalIO/Time'])        

    # get signal parameters
    time_sec = np.round(time_sec, 2)
    duration = max(time_sec)
    sampling_rate = int(len(isos) / duration)
    output_isos_min, output_isos_mean, output_isos_max = round(min(output_isos), 1), round(np.mean(output_isos), 1), round(max(output_isos), 1)
    output_fluo_min, output_fluo_mean, output_fluo_max = round(min(output_fluo), 1), round(np.mean(output_fluo), 1), round(max(output_fluo), 1)
    
    signal = pd.DataFrame({'Isos': isos, 'Fluo': fluo}, index=time_sec)   
    io = pd.DataFrame({'IO-1': digi_io}, index=digi_time)

    #io.to_csv(f'Jauuu.csv')

    # fuse Fluo, Isos and IO-1 datapoints (somehow signal starts at 0.1 and not 0) and delete rows with nans
    df = pd.concat([signal, io], axis=1)
    df.index.names = ['Time']
    df = df[['Fluo', 'Isos', 'IO-1']]
    df = df.dropna(subset=['Fluo', 'Isos'])

    # get TTL events from a digital channel
    events = event_list(df, 'IO-1')

    # output to check
    print(  f'LockIn\n'
            f'Output_Isos: {[output_isos_min, output_isos_mean, output_isos_max]}V, {len(output_isos)} dp\n'
            f'Output_Fluo: {[output_fluo_min, output_fluo_mean, output_fluo_max]}V, {len(output_fluo)} dp\n'
            f'Datapoints: {len(time_sec)} (Time), {len(isos)} (Isos), {len(fluo)} (Fluo), {len(digi_io)} (IO)\n'
            f'Duration: {duration}s, SamplingRate: {sampling_rate}Hz\n'
            f'Digital_IO: {len(events)}x {[ev[0] for ev in events]}\n')
    
    return df, events

def do_LockIn(file):

    with h5py.File(file, 'r') as f:
        path = 'DataAcquisition/NC500/Signals/Series0001/'

        # collect raw data from h5 file
        time_sec    = np.array(f[path + 'LockInAOUT01/Time'])
        isos        = np.array(f[path + 'LockInAOUT01/AIN01'])
        fluo        = np.array(f[path + 'LockInAOUT02/AIN01'])
        output_isos = np.array(f[path + 'AnalogOut/AOUT01'])
        output_fluo = np.array(f[path + 'AnalogOut/AOUT02'])
        

    # get signal parameters
    time_sec = np.round(time_sec, 2)
    duration = max(time_sec)
    sampling_rate = int(len(isos) / duration)
    output_isos_min, output_isos_mean, output_isos_max = round(min(output_isos), 1), round(np.mean(output_isos), 1), round(max(output_isos), 1)
    output_fluo_min, output_fluo_mean, output_fluo_max = round(min(output_fluo), 1), round(np.mean(output_fluo), 1), round(max(output_fluo), 1)
    
    df = pd.DataFrame({'Isos': isos, 'Fluo': fluo}, index=time_sec)   

    # fuse Fluo, Isos and IO-1 datapoints (somehow signal starts at 0.1 and not 0) and delete rows with nans
    df.index.names = ['Time']
    df = df[['Fluo', 'Isos']]
    df = df.dropna(subset=['Fluo', 'Isos'])



    # output to check
    print(  f'LockIn\n'
            f'Output_Isos: {[output_isos_min, output_isos_mean, output_isos_max]}V, {len(output_isos)} dp\n'
            f'Output_Fluo: {[output_fluo_min, output_fluo_mean, output_fluo_max]}V, {len(output_fluo)} dp\n'
            f'Datapoints: {len(time_sec)} (Time), {len(isos)} (Isos), {len(fluo)} (Fluo)\n'
            f'Duration: {duration}s, SamplingRate: {sampling_rate}Hz\n'
            )
    
    return df



def align(df, events, time_passed):

    # we have different amounts of time passed before the first event onset. Therefore subtract the same amount before first event
    # time_passed is the number of seconds that are found before the first event (do not take more than available)
    skip_time = round(events[0][0] - time_passed, 2)
    df.index = df.index - skip_time
    df = df.loc[0:]
    df.index = df.index.round(2)

    return df


def do(file):
    with h5py.File(file, 'r') as f:
        path = 'DataAcquisition/NC500/Signals/Series0001/'

        
        digi_io     = np.array(f[path + 'DigitalIO/DIO01'])
        digi_time   = np.array(f[path + 'DigitalIO/Time'])

    
    df = pd.DataFrame({'IO-1': digi_io}, index=digi_time)

    # fuse Fluo, Isos and IO-1 datapoints (somehow signal starts at 0.1 and not 0) and delete rows with nans
    
    

    # get TTL events from a digital channel
    events = event_list(df, 'IO-1')

    # output to check
    print(f'Digital_IO: {len(events)}x {[ev[0] for ev in events]}\n')
    
    return df, events


files = get_files(path)

liste = []

for file in files:

    print(f'\n{file}')

    df, events = do_LockIn_IO(file)
    #df = align(df, events, time_passed=0)    

    df = df[::10]



    # create data as .csv
    new_file = file.replace('.doric', '')
    title = f'{new_file}_LockIn_IO_10Hz'
    df.to_csv(f'{title}.csv')


#%%

print(io)
print(type(io[0]))

a = np.unique(io)
print(a)

transitions = np.sum((io[:-1] == 0.0) & (io[1:] == 1.0))

print("Number of segments of 1.0:", transitions)



