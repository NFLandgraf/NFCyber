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


#%%

path = r"D:\CA1Dopa_Miniscope\Post"
common_name = '.csv'

def get_files(path):
        # get files
        files = [os.path.join(path, file) for file in os.listdir(path) 
                if os.path.isfile(os.path.join(path, file)) and common_name in file]    
        
        print(f'{len(files)} files found')
        for file in files:
            print(f'{file}')
        print('\n')
        return files

files = get_files(path)


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
# collect, clean and fuse the traces& events
def split_traces_events(file_traces, file_events):
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

    def split_by_gap(df, gap_threshold=50):
    
        # take time_index, search gaps > threshold and define [start, endpoint] for splitting
        time_index = df.index.values
        gaps = np.diff(time_index)
        split_indices = np.where(gaps > gap_threshold)[0] + 1
        split_points = [0] + split_indices.tolist() + [len(df)]

        # go through timeseries and save each chunk defined by [start, end]
        individ_time_lengths = 0
        for i in range(len(split_points) - 1):
            start, end = split_points[i], split_points[i+1]
            chunk = df.iloc[start:end]
            chunk.index = (chunk.index - min(chunk.index.values)).round(1)
            chunk.to_csv(f"{path}\\chunk_part{i+1}.csv", index=True)
            individ_time_lengths += len(chunk.index.values)
        
        # checks that there is no mistake is splitting
        assert len(time_index) == individ_time_lengths, "split by gap diff index lengths"
        
        
    df_dff, df_z = get_traces(file_traces)
    df_events = get_traces_events(file_events, df_dff)
    main_df = fuse_df(df_dff, df_events)
    main_df.to_csv(f"{path}\\main_df.csv")

    split_by_gap(main_df)

    return main_df

file_traces = r"D:\CA1Dopa_Miniscope\Post\CA1Dopa_Longitud_f48_Post_traces.csv"
file_events = r"D:\CA1Dopa_Miniscope\Post\CA1Dopa_Longitud_f48_Post_events.csv"

main_df = split_traces_events(file_traces, file_events)








#%%
# collect, clean and fuse traces/events with DLC
def get_behav(file_DLC, file_TTL, file_dff, file_video, mirrorY=False):
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
        
        def clean_behav(file_DLC):
            # cleans the DLC file and returns the df
            # idea: instead of picking some 'trustworthy' points to mean e.g. head, check the relations between 'trustworthy' points, so when one is missing, it can gues more or less where it is
            bps_all = ['nose', 'left_ear', 'right_ear', 'left_ear_tip', 'right_ear_tip', 'left_eye', 'right_eye', 'head_midpoint', 
                        'neck', 'mid_back', 'mouse_center', 'mid_backend', 'mid_backend2', 'mid_backend3', 
                        'tail_base', 'tail1', 'tail2', 'tail3', 'tail4', 'tail5', 'tail_end',
                        'left_shoulder', 'left_midside', 'left_hip', 'right_shoulder', 'right_midside', 'right_hip']
            bps_head = ['nose', 'left_eye', 'right_eye', 'head_midpoint', 'left_ear', 'right_ear', 'neck']
            bps_postbody = ['mid_backend2', 'mid_backend3', 'tail_base']

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
                def get_video_dims(file_video):
                    cap = cv2.VideoCapture(file_video)
                    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()
                    print(f'{n_frames} video frames')
                    return (width, height)
                width, height = get_video_dims(file_video)
                for bp in bps_all:
                    df[f'{bp}_y'] = height - df[f'{bp}_y']
            
            # interpolate to skip nans
            df = df.interpolate(method="linear")
                                
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

            # check which row of behav is closest to the ISPX index and add these rows, all columns of df_behav to main_df
            behav_times = df_behav['Behav_Time'].values
            closest_behav_idx = np.abs(main_df.index.values[:, None] - behav_times).argmin(axis=1)
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

                df_behav['Behav_Time'] = TTLs_high_times
            
            return df_behav, TTLs_high_times

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
        df_behav = clean_behav(file_DLC)
        TTLs_high_times = clean_TTLs(file_TTL)

        # if TTLs are more than DLC rows
        if len(df_behav.index.values) == len(TTLs_high_times):
            df_behav['Behav_Time'] = TTLs_high_times
        else:
            df_behav, TTLs_high_times = different_length(df_behav, TTLs_high_times)
            
        main_df = fuse_df(file_dff, df_behav)
        # maybe delete Behav_time and Frame

        return main_df

data_prefix = "D:\\CA1Dopa_Miniscope\\Post\\CA1Dopa_Longitud_f48_Post3_YMaze_"

file_dff =   data_prefix + 'trace.csv'
file_DLC    =   data_prefix + 'DLC.csv'
file_TTL    =   data_prefix + 'GPIO.csv'
file_video  =   data_prefix + 'video_DLC.mp4'

main_df = get_behav(file_DLC, file_TTL, file_dff, file_video)
main_df.to_csv('main_df.csv')


#%%




#%%

data_prefix = r"D:\CA1Dopa_Miniscope\live"

file_traces =   data_prefix + 'traces.csv'
file_events =   data_prefix + 'events.csv'
file_DLC    =   data_prefix + 'DLC.csv'
file_TTL    =   data_prefix + 'GPIO.csv'
file_video  =   data_prefix + 'video_DLC.mp4'
file_tracesevents =   data_prefix + 'tracevents.csv'

files = [file_traces, file_events, file_DLC, file_TTL, file_video, file_tracesevents]
for f in files:
    if not Path(f).exists():
        print(f"❌ {f}")
    else:
        print(f"✅ {f}")



#%%

file_DLC    = r"D:\CA1Dopa_Miniscope\live\CA1Dopa_Longitud_f48_Post1_YMaze_DLC.csv"

