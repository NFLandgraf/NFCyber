#%%
# take the doric files and create a csv file with the completely raw signal and all IOs (choose number of IOs)
# if necessary, align everything to certain event(IO)

import h5py
import numpy as np
import pandas as pd
import os
# path = 'C:\\Users\\landgrafn\\Desktop\\2024-12-09_FF-Weilin-3m_Nomifensine\\Baseline\\'
path = 'C:\\Users\\landgrafn\\Desktop\\ffff\\'

common_name = '.doric'
rec_type = None
DLC_mm_per_px = 0.12

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

def do_LockIn_twoIO(file_doric):
    # do this when you have 2 IOs, e.g. IO2-Cam_TTL and IO3-FCbox

    with h5py.File(file_doric, 'r') as f:
        path = 'DataAcquisition/NC500/Signals/Series0001/'

        # collect raw data from h5 file
        sig_isos  = np.array(f[path + 'LockInAOUT01/AIN01'])
        sig_fluo  = np.array(f[path + 'LockInAOUT02/AIN01'])
        sig_time  = np.array(f[path + 'LockInAOUT01/Time'])

        digi_io2    = np.array(f[path + 'DigitalIO/DIO02']).astype(int)
        digi_io3    = np.array(f[path + 'DigitalIO/DIO03']).astype(int)
        digi_time   = np.array(f[path + 'DigitalIO/Time'])

        # output_isos = np.array(f[path + 'AnalogOut/AOUT01'])
        # output_fluo = np.array(f[path + 'AnalogOut/AOUT02'])
        # output_time = np.array(f[path + 'AnalogOut/Time'])

    # get signal parameters
    sig_time = np.round(sig_time, 2)
    duration = max(sig_time)
    sig_sampling_rate = int(len(sig_isos) / duration)
    signal = pd.DataFrame({'Isos': sig_isos, 'Fluo': sig_fluo}, index=sig_time)  
    # output_isos_min, output_isos_mean, output_isos_max = round(min(output_isos), 1), round(np.mean(output_isos), 1), round(max(output_isos), 1)
    # output_fluo_min, output_fluo_mean, output_fluo_max = round(min(output_fluo), 1), round(np.mean(output_fluo), 1), round(max(output_fluo), 1)
    
    # get IO parameters
    def extend_high_signal(df, column, extra_rows):
        # for one IO channel, extend the ocurrence of 1 to also see the high state when downsampling to 100Hz
        # Extend the 1s for x additional rows after a transition
        
        transitions = (df[column].shift(1) == 1) & (df[column] == 0)  # where does it change from 1 to 0
        for i in range(len(df)):
            if transitions.iloc[i]:  # If there was a transition from 1 to 0
                for j in range(0, extra_rows + 1):  # Add `1` for x extra rows
                    if i + j < len(df):  # Ensure we stay within bounds
                        df.iloc[i + j, df.columns.get_loc(column)] = 1
        return df
    io = pd.DataFrame({'IO2': digi_io2, 'IO3': digi_io3}, index=digi_time)
    io = io.loc[0::10]      # downsample to take every 10th frame -> one datapoint every ms
    io.index = io.index.round(3)
    io = extend_high_signal(io, 'IO2', 6)   # increase the duration to see every TTL at 100Hz

    # fuse Fluo, Isos and IO datapoints (somehow signal starts at 0.1 and not 0) and delete rows with nans
    df = pd.concat([signal, io], axis=1)
    df.index.names = ['Time']
    df = df[['Fluo', 'Isos', 'IO2', 'IO3']]
    df = df.dropna(subset=['Fluo', 'Isos'])

    # get TTL events from a digital channel
    events_IO2, events_IO3 = event_list(df, 'IO2'), event_list(df, 'IO3')

    # output
    print(  #f'Output_Isos: {[output_isos_min, output_isos_mean, output_isos_max]}V, {len(output_isos)} dp\n'
            #f'Output_Fluo: {[output_fluo_min, output_fluo_mean, output_fluo_max]}V, {len(output_fluo)} dp\n'
            f'Datapoints: Time {len(sig_time)}, Isos {len(sig_isos)}, Fluo {len(sig_fluo)}\n'
            f'Datapoints: IO2 {len(digi_io2)}, IO3 {len(digi_io3)}\n'
            f'Duration: {duration}s, SamplingRate: {sig_sampling_rate}Hz\n'
            f'Digital_IO2: {len(events_IO2)}x {[ev[0] for ev in events_IO2]}\n'
            f'Digital_IO3: {len(events_IO3)}x {[ev[0] for ev in events_IO3]}')
    
    return df, events_IO2, events_IO3

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

def get_distances(file_doric, bps_position='mid_backend', min_distance=0.001):
    # calculates the distance travelled between frames, where 

    file_DLC = file_doric.replace(common_name, '_DLC.csv')

    def cleaning_raw_df(file_DLC):

        df = pd.read_csv(file_DLC, header=None, low_memory=False)

        # create new column names from first two rows
        new_columns = [f"{col[0]}_{col[1]}" for col in zip(df.iloc[1], df.iloc[2])]
        df.columns = new_columns
        df = df.drop(labels=[0, 1, 2], axis="index")

        # set index and convert to numeric
        df.set_index('bodyparts_coords', inplace=True)
        df.index.names = ['frames']
        df = df.astype(float)
        df.index = df.index.astype(int)

        bps_all = ['nose', 'left_ear', 'right_ear', 'left_ear_tip', 'right_ear_tip', 'left_eye', 'right_eye', 'head_midpoint', 
                 'neck', 'mid_back', 'mouse_center', 'mid_backend', 'mid_backend2', 'mid_backend3', 
                 'tail_base', 'tail1', 'tail2', 'tail3', 'tail4', 'tail5', 'tail_end',
                 'left_shoulder', 'left_midside', 'left_hip', 'right_shoulder', 'right_midside', 'right_hip']

        # remove low-confidence points
        for bodypart in bps_all:
            filter = df[f'{bodypart}_likelihood'] <= 0.9
            df.loc[filter, f'{bodypart}_x'] = np.nan
            df.loc[filter, f'{bodypart}_y'] = np.nan
            df = df.drop(columns=f'{bodypart}_likelihood')

        return df
    
    def euclidian_dist(point1x, point1y, point2x, point2y):
        # calculates distance between 2 points in carthesic coordinate system
        return np.sqrt((point1x - point2x) ** 2 + (point1y - point2y) ** 2)

    df = cleaning_raw_df(file_DLC)
    x_col, y_col = f'{bps_position}_x', f'{bps_position}_y'
    
    # creates a full index range (to include missing frames), fill Series with NaNs
    full_index = np.arange(df.index.min(), df.index.max() + 1)
    dist_series = pd.Series(index=full_index, dtype=float)

    # goes though all frames/rows and calcs distance between the frames with values (not NaN)
    last_valid_idx = None
    for i in df.index:
        if not np.isnan(df.at[i, x_col]) and not np.isnan(df.at[i, y_col]):
            if last_valid_idx is not None:
                dist = euclidian_dist(df.at[last_valid_idx, x_col], df.at[last_valid_idx, y_col], df.at[i, x_col], df.at[i, y_col])
                dist_series[i] = dist if dist >= min_distance else np.nan  
            last_valid_idx = i
    dist_travelled = round(dist_series.sum(), 1)

    # goes through the frames and distributes the distance evenly across NaNs
    nan_indices = np.where(dist_series.isna())[0]
    if len(nan_indices) > 0:
        last_valid_idx = None
        for i in range(len(dist_series)):
            if not np.isnan(dist_series[i]):  
                if last_valid_idx is not None:
                    num_missing = i - last_valid_idx - 1  # Frames between valid distances
                    if num_missing > 0:
                        total_dist = dist_series[i]  # Distance to be spread
                        per_frame_dist = total_dist / (num_missing + 1)
                        for j in range(1, num_missing + 2):
                            dist_series[last_valid_idx + j] = per_frame_dist
                last_valid_idx = i
    dist_series[0] = 0.0
    dist_travelled_distr = round(dist_series.sum(), 1)
    if dist_travelled != dist_travelled_distr:
        print(f'Error: different distances calculated: {dist_travelled} vs {dist_travelled_distr}')
    
    # convert to mm and to DataFrame
    dist = dist * DLC_mm_per_px
    df_distances = dist_series.to_frame(name=f'{bps_position}_dist')
    # print(f'distance travelled: {round(df_distances.sum(), 1)}mm')
    
    # df_distances.to_csv(f'{file_DLC}_distances.csv')

    return df_distances
def add_distances_to_maindf(file_doric, main_df, camTTL_column='IO2'):
    # Take a column from the main_df and the df of the movements between frames
    # Now in which row here is a one in the Cam_TTL_column, add the corresponding travel distance

    df_distances = get_distances(file_doric)
    timepoint_frame = main_df.index[main_df[camTTL_column] == 1]    

    # check that Cam_frames and Cam_TTL are the same
    nb_frames_TTL = len(timepoint_frame)
    nb_frames_DLC = df_distances.shape[0]
    print(f'nb_frames_DLC/nb_frames_TTL -> {nb_frames_DLC}/{nb_frames_TTL}')

    # if the number of TTL and frames dont mach up, just add some more distances/frames with distance=0
    while nb_frames_DLC != nb_frames_TTL:
        print(f'{nb_frames_DLC}/{nb_frames_TTL}')
        df_distances.loc[len(df_distances), 'mid_backend_dist'] = 0
        nb_frames_DLC = df_distances.shape[0]

    # add the distances to the main dataframe, whereever there is a 1 at the respective IO
    main_df['Distances'] = np.nan
    main_df.loc[timepoint_frame, 'Distances'] = df_distances.iloc[:len(timepoint_frame)].values.flatten()

    return main_df


files_doric = get_files(path)
for file_doric in files_doric:
    print(f'\n{file_doric}')

    # if 2 IOs
    main_df, events_IO2, events_IO3 = do_LockIn_twoIO(file_doric)
    main_df = add_distances_to_maindf(file_doric, main_df)
    main_df = align_to_events(main_df, events_IO3)    

    # if 1 IOs
    # df, events_IO = do_LockIn_oneIO(file)
    # df = align(df, events_IO)

    # if 0 IOs
    # df = do_LockIn(file)
    # df = crop_recording_no_events(df)

    new_file = file_doric.replace('.doric', '_LockIn.csv')
    print(new_file)
    main_df.to_csv(new_file)


