#%%
# IMPORT
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import csv
from pathlib import Path
import os
from scipy.stats import zscore



#%%

def activity_heatmap(df):
    df_t = df.T

    plt.imshow(df_t, aspect='auto', cmap='seismic', vmin=-3, vmax=3)
    plt.colorbar(label='z-score')

    tick_positions = range(0, len(df_t.columns), 500)
    plt.xticks(ticks=tick_positions, labels=df_t.columns[tick_positions], rotation=90)

    plt.xlabel('Time [s]')
    plt.ylabel('Cells')
    plt.title('Activity Heatmap')
    plt.show()
    plt.savefig('heatmap.pdf', dpi=300)
    plt.close()



#%%
'''
So, for this code we have a series of recordings, e.g. OF+YMaze together from the same day recording or recordings from sveral days, that were analyzed together in IDPS
When exported from IDPS, these recording chunks are separated by gaps in time_index (at least 10s I think, but check every time)
We use those gaps to divide the chunks to get one chunk per recording
1. clean the file_traces blablabla and return the dff files, cells are columns
2. clean the file_events and digitize the spikes, then add the 0 or 1 to the dff df
3. split the file into individual recording chunks according to the time threshold
'''
def split_traces_events(file_traces, file_events, data_prefix):
    # use this when you have a time series of several recordings (e.g. Baseline, 15Hz, etc. in one time series)
    # this gets traces and events and puts them together, then it splits the time series into individual ones and saves them
    def get_traces(file_traces):
        # clean the csv file and return df
        df = pd.read_csv(file_traces, low_memory=False)

        # drop rejected cells/columns
        rejected_columns = df.columns[df.isin([' rejected']).any()]
        df = df.drop(columns=rejected_columns)

        # Clean up df
        df = df.drop(0)
        df = df.astype(float)
        df = df.rename(columns = {' ': 'Time'})
        df = df.set_index(['Time'])
        df.index = df.index.round(1)

        # remove the C, remove the 001 zeros and turn it back into str
        df.columns = df.columns.str.replace(' C', '')
        df.columns = df.columns.astype(int)
        df.columns = df.columns.astype(str)

        # change a little bit the dff trace
        df_dff = df.copy()
        df_dff.columns = 'dff_' + df_dff.columns

        # normalize it via zscore
        df_z = df.copy()
        df_z.columns = 'z_' + df_z.columns
        for name, data in df_z.items():
            df_z[name] = (df_z[name] - df_z[name].mean()) / df_z[name].std()    

        return df_dff, df_z

    def get_traces_events(file_events, df_dff, rounding=1):

        # create df filled with zeros with df_dff index
        df_raw = pd.read_csv(file_events, low_memory=False)
        df_events = pd.DataFrame(0, index=df_dff.index, columns=sorted(df_raw[" Cell Name"].unique()))

        # go through event csv and add every value into df filled with zeros
        for _, row in df_raw.iterrows():
            time = round(row["Time (s)"], rounding)
            cell = row[" Cell Name"]
            if time in df_events.index:
                df_events.at[time, cell] = 1
            else:
                print(time, cell)

        # rename columns accordingly
        df_events.columns = df_events.columns.str.replace('C', '').astype(int)
        df_events.columns = 'spike_' + df_events.columns.astype(str)

        # digitize the df (0-nospike, 1-spike) and forget the real values
        #df_events = (df_events != 0).astype(int)

        return df_events

    def fuse_df(df_dff, df_events):
        

        assert df_dff.index.equals(df_events.index), "Df indices no match!"
        main_df = pd.concat([df_dff, df_events], axis=1)
        return main_df

    def split_by_gap(df, gap_threshold=10):
    
        # take time_index, search gaps > threshold and define [start, endpoint] for splitting
        time_index = df.index.values
        gaps = np.diff(time_index)
        split_indices = np.where(gaps > gap_threshold)[0] + 1
        split_points = [0] + split_indices.tolist() + [len(df)]

        # sanity check
        gaps = np.round(gaps, 3)
        unique_vals, counts = np.unique(gaps, return_counts=True)
        print(f'TTL gaps: {[float(x) for x in unique_vals]}\nTTL numb: {counts}')

        # go through timeseries and save each chunk defined by [start, end]
        individ_time_lengths = 0
        for i in range(len(split_points) - 1):
            start, end = split_points[i], split_points[i+1]
            chunk = df.iloc[start:end]
            chunk.index = (chunk.index - min(chunk.index.values)).round(1)
            chunk.to_csv(f"{data_prefix}chunk_part{i}.csv", index=True)
            individ_time_lengths += len(chunk.index.values)
        
        # checks that there is no mistake is splitting
        assert len(time_index) == individ_time_lengths, "split by gap diff index lengths"
        
        
    df_dff, df_z = get_traces(file_traces)
    df_events = get_traces_events(file_events, df_dff)
    main_df = fuse_df(df_dff, df_events)
    #main_df.to_csv(f"{data_prefix}main_nosplit.csv")

    split_by_gap(main_df)

    return main_df

data_prefix = "D:\\CA1Dopa_Miniscope\\Post_f51\\CA1Dopa_Longitud_f51_Post_"
file_traces = data_prefix + 'traces.csv'
file_events = data_prefix + 'events.csv'

main_df = split_traces_events(file_traces, file_events, data_prefix)






#%%
'''
So, this code implements the csv_DLC file, the ISPX_TTL file, the DLC_video file and a csv with dff/events traces from ISPX
1. cleans the TTL file and gets the rising times from DAQ (when a TTL was sent to trigger a frame or when DAQ received a TTL from a recorded frame)
2. cleans the DLC behavior csv with the frames as index and all the bodyparts together with head and postbody
3. adds the rising DAQ times to the behav_file, so we have the frames and the corresponding DAQ master clock times
4. takes the traces/dff csv with original DAQ times that was created and cleaned earlier
5. takes the index of the traces file (original DAQ times) and searches for the closest corresponding TTLs in the behav_df, then fuses the respective behav rows to the traces_df
6. trims the resulting main_df depending on the video frames where the animal is released/taken out into/out of the area
7. normalizes the dff of each cell of the final df via zscore and saves the resulting file as csv
'''
def get_behav(file_DLC, file_TTL, file_dff, file_video, trim_frames, data_prefix, mirrorY=False):
        # if you filmed, you will have a video file (DLC file) and TTLs in the GPIO file
        # this def gets the DLC animal position and synchronizes it with the Inspx time

        def clean_TTLs(file_TTL, channel_name = ' GPIO-1', value_threshold=10000, constant=True):
            # returns list of TTL input times

            # expects nan values to be just a string: ' nan', convert these strings into proper nans
            df = pd.read_csv(file_TTL, na_values=[' nan'], low_memory=False)
            df = df.set_index('Time (s)')

            # only cares for the rows where the channel name is in the ' Channel Name' column
            df = df[df[' Channel Name'] == channel_name]
            
            # drops rows with nan and converts everything into ints
            df = df.dropna(subset=[' Value'])
            df[' Value'] = df[' Value'].astype(int)

            # creates series with True for values > threshold
            # checks at which index/time the row's value is high AND the row before's value is low
            # for that, it makes the whole series negative (True=False) and shifts the whole series one row further (.shift)
            # so, onset wherever row is > threshold and the opposite value of the shifted (1) row is positive as well
            is_high = df[' Value'] >= value_threshold
            onsets = df.index[is_high & ~is_high.shift(1, fill_value=False)]
            offsets = df.index[~is_high & is_high.shift(1, fill_value=False)]

            # high times are all the times where signal goes from below to above threshold
            high_times = list(zip(onsets, offsets)) if not constant else list(onsets)

            # sanity check
            high_times_arr = np.array(high_times)
            high_times_intervals = np.round(np.diff(high_times_arr), 5)
            unique_vals, counts = np.unique(high_times_intervals, return_counts=True)
            if len(unique_vals) > 1:
                print('GPIO TTL intervals are not uniform')
                print(f'unique_valus: {unique_vals}')
                print(f'unique_count: {counts}')

            print(f'{len(high_times)} TTLs')
            return high_times
        
        def clean_behav(file_DLC, file_video):
            # cleans the DLC file and returns the df
            # idea: instead of picking some 'trustworthy' points to mean e.g. head, check the relations between 'trustworthy' points, so when one is missing, it can gues more or less where it is
            bps_all = ['nose', 'left_ear', 'right_ear', 'left_ear_tip', 'right_ear_tip', 'left_eye', 'right_eye', 'head_midpoint', 
                        'neck', 'mid_back', 'mouse_center', 'mid_backend', 'mid_backend2', 'mid_backend3', 
                        'tail_base', 'tail1', 'tail2', 'tail3', 'tail4', 'tail5', 'tail_end',
                        'left_shoulder', 'left_midside', 'left_hip', 'right_shoulder', 'right_midside', 'right_hip']
            bps_head = ['nose', 'left_eye', 'right_eye', 'head_midpoint', 'left_ear', 'right_ear', 'neck']
            bps_postbody = ['mid_backend2', 'mid_backend3', 'tail_base']

            def get_video_dims(file_video):
                cap = cv2.VideoCapture(file_video)
                n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                return (width, height)
            width, height = get_video_dims(file_video)

            df = pd.read_csv(file_DLC, header=None, low_memory=False)

            # combines the string of the two rows to create new header
            # df.iloc[0] takes all values of first row
            new_columns = [f"{col[0]}_{col[1]}" for col in zip(df.iloc[1], df.iloc[2])]     
            df.columns = new_columns
            df = df.drop(labels=[0, 1, 2], axis="index")

            # adapt index and turn all the prior string values into floats and index into int
            df.set_index('bodyparts_coords', inplace=True)
            df.index.names = ['Frame']
            df = df.astype(float)
            df.index = df.index.astype(int)

            # insert NaN if likelihood is below x
            for bp in bps_all:
                filter = df[f'{bp}_likelihood'] <= 0.9     
                df.loc[filter, f'{bp}_x'] = np.nan
                df.loc[filter, f'{bp}_y'] = np.nan
                df = df.drop(labels=f'{bp}_likelihood', axis="columns")
            
            if mirrorY:
                # mirror y and forget everything except head and postbody
                for bp in bps_all:
                    df[f'{bp}_y'] = height - df[f'{bp}_y']
            
            # interpolate to skip nans
            df = df.interpolate(method="linear")


            # transform the coordinates into mm to normalize for camera reposition between recordings (we have the same dimension in mm during video_processing so all good)
            if 'OF' in data_prefix:
                scale_x = 460 / width   # mm/px
                scale_y = 460 / height  # mm/px
                print('OF! for transformation px into mm ')
            elif 'YMaze' in data_prefix:
                scale_x = 580 / width   # mm/px
                scale_y = 485 / height  # mm/px
                print('YMaze! for transformation px into mm ')
            else:
                raise ValueError('Bro, Transformation px into mm doesnt know which arena you use')
            
            for col in df.columns:
                if col.endswith('_x'):
                    df[col] = df[col] * scale_x
                elif col.endswith('_y'):
                    df[col] = df[col] * scale_y
 

            # mean 'trustworthy' bodyparts (ears, ear tips, eyes, midpoint, neck) to 'head' column
            for c in ('_x', '_y'):
                df[f'head{c}'] = df[[bp+c for bp in bps_head]].mean(axis=1, skipna=True)
                df[f'postbody{c}'] = df[[bp+c for bp in bps_postbody]].mean(axis=1, skipna=True)

            # # make one column (x,y) per bodypart
            # bodyparts = sorted(set(c[:-2] for c in df.columns if c.endswith(("_x", "_y"))))
            # df_behav = pd.DataFrame(index=df.index)
            # for bp in bodyparts:
            #     df_behav[bp] = list(zip(df[f"{bp}_x"], df[f"{bp}_y"]))

            print(f'{df.shape[0]} DLC frames')
            return df
        
        def fuse_df(file_dff, df_behav):

            main_df = pd.read_csv(file_dff, index_col='Time', low_memory=False)

            # adds the index as a regular column to keep it after the fusion
            df_behav['Frame'] = df_behav.index

            # check which row of behav is closest to the ISPX index and add these rows, all columns of df_behav to main_df
            TTL_times = df_behav['TTL_Time'].values
            closest_behav_idx = np.abs(main_df.index.values[:, None] - TTL_times).argmin(axis=1)
            main_df[df_behav.columns] = df_behav.iloc[closest_behav_idx].values

            return main_df

        def different_length(df_behav, TTLs_high_times):

            #assert len(df_behav.index.values) == len(TTLs_high_times), f'TTL ({len(TTLs_high_times)}) and DLC_frames ({len(df_behav.index.values)}) different'
            print(f'WARNING: TTL ({len(TTLs_high_times)}) and DLC_frames ({len(df_behav.index.values)}) different')
            missing = len(TTLs_high_times) - len(df_behav)
            if missing > 0:
                # Create DataFrame with NaNs for missing rows
                df_missing = pd.DataFrame(np.nan, index=range(len(df_behav), len(df_behav)+missing), columns=df_behav.columns)
                df_behav = pd.concat([df_behav, df_missing])

                df_behav['TTL_Time'] = TTLs_high_times
            
            return df_behav, TTLs_high_times

        def trim_n_norm(df, trim_frames):

            start_frame, stop_frame = trim_frames  # allow None
            
            # trim the df according to start and stop frame
            mask = df['Frame'] >= (float(start_frame) if start_frame is not None else df['Frame'].min())
            if stop_frame is not None:
                mask &= df['Frame'] <= float(stop_frame)

            
            df_trimmed = df.loc[mask].copy()
            df_trimmed.index = (df_trimmed.index - df_trimmed.index[0]).round(1)


            # normalize via zscore
            dff_cols = [col for col in df_trimmed.columns if 'dff' in col]
            df_z = (df_trimmed[dff_cols] - df_trimmed[dff_cols].mean()) / df_trimmed[dff_cols].std()
            df_z.columns = [col.replace('dff', 'z') for col in df_z.columns]
            df = pd.concat([df_trimmed, df_z], axis=1)

            # arrange the columns
            other_cols = [c for c in df.columns if all(x not in c for x in ['spike', 'z', 'dff', 'Frame', 'TTL_Time'])]
            df = df[['Frame', 'TTL_Time'] + [c for c in df.columns if 'spike' in c] + [c for c in df.columns if 'z' in c] + [c for c in df.columns if 'dff' in c] + other_cols]

            return df

        def create_video(df):
           
            # Your bodyparts
            bps_all = ['nose', 'left_ear', 'right_ear', 'left_ear_tip', 'right_ear_tip', 
                    'left_eye', 'right_eye', 'head_midpoint', 'neck', 'mid_back', 
                    'mouse_center', 'mid_backend', 'mid_backend2', 'mid_backend3', 
                    'tail_base', 'tail1', 'tail2', 'tail3', 'tail4', 'tail5', 'tail_end',
                    'left_shoulder', 'left_midside', 'left_hip', 'right_shoulder', 
                    'right_midside', 'right_hip']


            # Video parameters
            frame_width = 650
            frame_height = 550
            fps = 30
            video_name = "bodyparts_video.mp4"

            # Create VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_name, fourcc, fps, (frame_width, frame_height))

            # Loop through frames
            for idx, row in df.iterrows():
                # Create black canvas
                frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                
                # Draw each bodypart
                for bp in bps_all:
                    x_col = f"{bp}_x"
                    y_col = f"{bp}_y"
                    
                    if x_col in df.columns and y_col in df.columns:
                        x_val = row[x_col]
                        y_val = row[y_col]
                        
                        # Skip NaN coordinates
                        if np.isnan(x_val) or np.isnan(y_val):
                            continue

                        x = int(x_val)
                        y = int(y_val)
                        
                        # Draw circle
                        cv2.circle(frame, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
                
                # Write frame to video
                out.write(frame)

            # Release VideoWriter
            out.release()
            print(f"Video saved as {video_name}")

        # get the files and use the TTL list as the index for behav
        df_behav = clean_behav(file_DLC, file_video)
        TTLs_high_times = clean_TTLs(file_TTL)

        # if TTLs are more than DLC rows
        if len(df_behav.index.values) == len(TTLs_high_times):
            df_behav['TTL_Time'] = TTLs_high_times
        else:
            df_behav, TTLs_high_times = different_length(df_behav, TTLs_high_times)
        
        # fuse both df
        main_df = fuse_df(file_dff, df_behav)

        # arrange the columns
        other_cols = [c for c in main_df.columns if all(x not in c for x in ['spike', 'z', 'dff', 'Frame', 'TTL_Time'])]
        main_df = main_df[['Frame', 'TTL_Time'] + [c for c in main_df.columns if 'spike' in c] + [c for c in main_df.columns if 'z' in c] + [c for c in main_df.columns if 'dff' in c] + other_cols]

        # trim and normalize the trimmed df
        main_df = trim_n_norm(main_df, trim_frames)

        main_df.to_csv(data_prefix + 'main.csv')
        return main_df

data_prefix = "D:\\CA1Dopa_Miniscope\\Post_f48\\CA1Dopa_Longitud_f48_Post3_YMaze_"
file_dff    =   data_prefix + 'trace.csv'
file_DLC    =   data_prefix + 'DLC.csv'
file_TTL    =   data_prefix + 'GPIO.csv'
file_video  =   data_prefix + 'video_DLC.mp4'
trim_frames = (766, None)

main_df = get_behav(file_DLC, file_TTL, file_dff, file_video, trim_frames, data_prefix)


