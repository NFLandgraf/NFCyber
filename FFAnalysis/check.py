#%%
# Take the doric files and create a csv file with the completely raw signal and IOs

import h5py
import numpy as np
import pandas as pd
import os
# path = 'C:\\Users\\landgrafn\\Desktop\\2024-12-09_FF-Weilin-3m_Nomifensine\\Baseline\\'
path = 'C:\\Users\\landgrafn\\Desktop\\BaselineCheck\\'

common_name = '.doric'
rec_type = None

def h5print(item, leading=''):
    # prints out the h5 data structure
    for key in item:
        if isinstance(item[key], h5py.Dataset):
            print(leading + key + ':' + str(item[key].shape))
        else:
            print(leading + key)
            h5print(item[key], leading + '  ')
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
def do_LockIn(file):

    with h5py.File(file, 'r') as f:
        # h5print(f, '')

        path = 'DataAcquisition/NC500/Signals/Series0001/'

        # collect raw data from h5 file
        time_sec    = np.array(f[path + 'LockInAOUT01/Time'])
        isos        = np.array(f[path + 'LockInAOUT01/AIN01'])
        fluo        = np.array(f[path + 'LockInAOUT02/AIN01'])
        #output_isos = np.array(f[path + 'AnalogOut/AOUT01'])
        #output_fluo = np.array(f[path + 'AnalogOut/AOUT02'])
        

    # get signal parameters
    time_sec = np.round(time_sec, 2)
    duration = max(time_sec)
    sampling_rate = int(len(isos) / duration)
    #output_isos_min, output_isos_mean, output_isos_max = round(min(output_isos), 1), round(np.mean(output_isos), 1), round(max(output_isos), 1)
    #output_fluo_min, output_fluo_mean, output_fluo_max = round(min(output_fluo), 1), round(np.mean(output_fluo), 1), round(max(output_fluo), 1)
    
    df = pd.DataFrame({'Isos': isos, 'Fluo': fluo}, index=time_sec)   

    # fuse Fluo, Isos and IO-1 datapoints (somehow signal starts at 0.1 and not 0) and delete rows with nans
    df.index.names = ['Time']
    df = df[['Fluo', 'Isos']]
    df = df.dropna(subset=['Fluo', 'Isos'])



    # output to check
    print(  f'LockIn\n'
            #'Output_Isos: {[output_isos_min, output_isos_mean, output_isos_max]}V, {len(output_isos)} dp\n'
            #f'Output_Fluo: {[output_fluo_min, output_fluo_mean, output_fluo_max]}V, {len(output_fluo)} dp\n'
            f'Datapoints: {len(time_sec)} (Time), {len(isos)} (Isos), {len(fluo)} (Fluo)\n'
            f'Duration: {duration}s, SamplingRate: {sampling_rate}Hz\n'
            )
    
    return df
def do_LockIn_oneIO(file):

    with h5py.File(file, 'r') as f:
        h5print(f, '')
        path = 'DataAcquisition/NC500/Signals/Series0001/'

        # collect raw data from h5 file
        sig_isos  = np.array(f[path + 'LockInAOUT01/AIN01'])
        sig_fluo  = np.array(f[path + 'LockInAOUT02/AIN01'])
        sig_time  = np.array(f[path + 'LockInAOUT01/Time'])

        # output_isos = np.array(f[path + 'AnalogOut/AOUT01'])
        # output_fluo = np.array(f[path + 'AnalogOut/AOUT02'])
        # output_time = np.array(f[path + 'AnalogOut/Time'])

        digi_io    = np.array(f[path + 'DigitalIO/DIO01']).astype(int)
        
        digi_time   = np.array(f[path + 'DigitalIO/Time'])        

    # get signal parameters
    sig_time = np.round(sig_time, 2)
    duration = max(sig_time)
    sig_sampling_rate = int(len(sig_isos) / duration)
    # output_isos_min, output_isos_mean, output_isos_max = round(min(output_isos), 1), round(np.mean(output_isos), 1), round(max(output_isos), 1)
    # output_fluo_min, output_fluo_mean, output_fluo_max = round(min(output_fluo), 1), round(np.mean(output_fluo), 1), round(max(output_fluo), 1)
    
    signal = pd.DataFrame({'Isos': sig_isos, 'Fluo': sig_fluo}, index=sig_time)   
    io = pd.DataFrame({'IO': digi_io}, index=digi_time)

    # fuse Fluo, Isos and IO datapoints (somehow signal starts at 0.1 and not 0) and delete rows with nans
    df = pd.concat([signal, io], axis=1)
    df.index.names = ['Time']
    df = df[['Fluo', 'Isos', 'IO']]
    df = df.dropna(subset=['Fluo', 'Isos'])

    # get TTL events from a digital channel
    events_IO = event_list(df, 'IO')
    

    # output to check
    print(  f'LockIn\n'
            #f'Output_Isos: {[output_isos_min, output_isos_mean, output_isos_max]}V, {len(output_isos)} dp\n'
            #f'Output_Fluo: {[output_fluo_min, output_fluo_mean, output_fluo_max]}V, {len(output_fluo)} dp\n'
            f'Datapoints: {len(sig_time)} (Time), {len(sig_isos)} (Isos), {len(sig_fluo)} (Fluo)\n'
            f'Datapoints: {len(digi_io)} (IO2)\n'
            f'Duration: {duration}s, SamplingRate: {sig_sampling_rate}Hz\n'
            f'Digital_IO2: {len(events_IO)}x {[ev[0] for ev in events_IO]}\n')
    
    return df, events_IO
def do_LockIn_twoIO(file):

    with h5py.File(file, 'r') as f:
        # h5print(f, '')
        path = 'DataAcquisition/NC500/Signals/Series0001/'

        # collect raw data from h5 file
        sig_isos  = np.array(f[path + 'LockInAOUT01/AIN01'])
        sig_fluo  = np.array(f[path + 'LockInAOUT02/AIN01'])
        sig_time  = np.array(f[path + 'LockInAOUT01/Time'])

        # output_isos = np.array(f[path + 'AnalogOut/AOUT01'])
        # output_fluo = np.array(f[path + 'AnalogOut/AOUT02'])
        # output_time = np.array(f[path + 'AnalogOut/Time'])

        digi_io2    = np.array(f[path + 'DigitalIO/DIO02']).astype(int)
        digi_io3    = np.array(f[path + 'DigitalIO/DIO03']).astype(int)
        digi_time   = np.array(f[path + 'DigitalIO/Time'])        

    # get signal parameters
    sig_time = np.round(sig_time, 2)
    duration = max(sig_time)
    sig_sampling_rate = int(len(sig_isos) / duration)
    # output_isos_min, output_isos_mean, output_isos_max = round(min(output_isos), 1), round(np.mean(output_isos), 1), round(max(output_isos), 1)
    # output_fluo_min, output_fluo_mean, output_fluo_max = round(min(output_fluo), 1), round(np.mean(output_fluo), 1), round(max(output_fluo), 1)
    
    signal = pd.DataFrame({'Isos': sig_isos, 'Fluo': sig_fluo}, index=sig_time)   
    io = pd.DataFrame({'IO2': digi_io2, 'IO3': digi_io3}, index=digi_time)

    # fuse Fluo, Isos and IO datapoints (somehow signal starts at 0.1 and not 0) and delete rows with nans
    df = pd.concat([signal, io], axis=1)
    df.index.names = ['Time']
    df = df[['Fluo', 'Isos', 'IO2', 'IO3']]
    df = df.dropna(subset=['Fluo', 'Isos'])

    # get TTL events from a digital channel
    events_IO2 = event_list(df, 'IO2')
    events_IO3 = event_list(df, 'IO3')

    # output to check
    print(  f'LockIn\n'
            #f'Output_Isos: {[output_isos_min, output_isos_mean, output_isos_max]}V, {len(output_isos)} dp\n'
            #f'Output_Fluo: {[output_fluo_min, output_fluo_mean, output_fluo_max]}V, {len(output_fluo)} dp\n'
            f'Datapoints: {len(sig_time)} (Time), {len(sig_isos)} (Isos), {len(sig_fluo)} (Fluo)\n'
            f'Datapoints: {len(digi_io2)} (IO2), {len(digi_io3)} (IO3)\n'
            f'Duration: {duration}s, SamplingRate: {sig_sampling_rate}Hz\n'
            f'Digital_IO2: {len(events_IO2)}x {[ev[0] for ev in events_IO2]}\n'
            f'Digital_IO3: {len(events_IO3)}x {[ev[0] for ev in events_IO3]}\n')
    
    return df, events_IO2, events_IO3, sig_sampling_rate

def align_to_events(df, events, time_passed_pre=30, time_passed_post=30):

    # we have different amounts of time passed before the first event onset and after the last event onset. 
    # Therefore, delete anything time_passed_post s after last event and subtract time_passed_pre s before first event
    # time_passed_pre/_post is the number of seconds before/after the first/last event (do not take more than available)
    skip_time_pre   = round(events[0][0] - time_passed_pre, 2)
    skip_time_post  = round(events[-1][0] + time_passed_post, 2)

    # delete last
    df = df.loc[:skip_time_post]

    # delete first
    df.index = df.index - skip_time_pre
    df = df.loc[0:]
    df.index = df.index.round(2)

    return df

def crop_recording_no_events(df, length=15.0, start=15.0):

    # we have a recording without events and want to have it at a certain length at a certain time
    end = start + length
    df = df.loc[start : end]


    return df



files = get_files(path)
for file in files:
    print(f'\n{file}')

    # if 2 IOs
    df, events_IO2, events_IO3, sig_sampling_rate = do_LockIn_twoIO(file)
    df = align_to_events(df, events_IO3)    

    # if 1 IOs
    # df, events_IO = do_LockIn_oneIO(file)
    # df = align(df, events_IO)

    # if 0 IOs
    # df = do_LockIn(file)
    # df = crop_recording_no_events(df)



    # GuPPy
    df.loc[0, 'sampling_rate'] = sig_sampling_rate
    del df['IO2']
    del df['IO3']
    del df['Fluo']
    df.index.names = ['timestamps']
    df.rename(columns={'Isos': 'data'}, inplace=True)

    print(df)




    # create data as .csv
    new_file = file.replace('.doric', '')
    title = f'{new_file}_LockIn'
    df.to_csv(f'{title}.csv')


