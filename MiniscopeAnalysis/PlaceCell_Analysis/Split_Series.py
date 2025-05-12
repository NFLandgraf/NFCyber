#%%
# IMPORT and COMBINE
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

data_prefix = 'C:\\Users\\landgrafn\\Desktop\\eyy\\CA1Dopa_Longitud_m32_Week2_'

file_traces =   data_prefix + 'traces.csv'
file_events =   data_prefix + 'events.csv'
file_DLC    =   data_prefix + 'DLC.csv'
file_TTL    =   data_prefix + 'GPIO.csv'
file_video  =   data_prefix + 'video.mp4'
file_tracesevents =   data_prefix + 'traces+events.csv'

files = [file_traces, file_events, file_DLC, file_TTL, file_video, file_tracesevents]
for f in files:
    if not Path(f).exists():
        print(f"❌ {f}")
    else:
        print(f"✅ {f}")
    

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
        df['Time'] = df['Time'].round(1)
        df = df.set_index(['Time'])
        df.index = df.index.round(2)

        df.columns = df.columns.str.replace(' C', '')
        df.columns = df.columns.astype(int)
        df_dff = df

        # normalize it via zscore
        df_z = df.copy()
        for name, data in df_z.items():
            df_z[name] = (df_z[name] - df_z[name].mean()) / df_z[name].std()    

        #activity_heatmap(df_z)

        #df_z = df_z[[175]]

        return df_dff, df_z

    def get_traces_events(file_events, df_z, rounding=1):

        # create df filled with zeros with df_z index
        df_raw = pd.read_csv(file_events, low_memory=False)
        df_events = pd.DataFrame(0, index=df_z.index, columns=sorted(df_raw[" Cell Name"].unique()))

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
        df_events.columns = df_events.columns.astype(str) + '_spike'

        # digitize the df (0-nospike, 1-spike) and forget the real values
        df_events = (df_events != 0).astype(int)

        #df_events = df_events[['175_spike']]

        # merge the new columns to df_z
        assert df_z.index.equals(df_events.index), "Indices do not match!"
        df_combined = pd.concat([df_z, df_events], axis=1)

        return df_combined

    def split_by_gap(df, gap_threshold=50):
    
        # take time_index, search gaps > threshold and define [start, endpoint] for splitting
        time_index = df.index.values
        gaps = np.diff(time_index)
        split_indices = np.where(gaps > gap_threshold)[0] + 1
        split_points = [0] + split_indices.tolist() + [len(df)]

        # go through timeseries and save each chunk defined by [start, end]
        for i in range(len(split_points) - 1):
            start, end = split_points[i], split_points[i+1]
            chunk = df.iloc[start:end]
            chunk.index = (chunk.index - min(chunk.index.values)).round(1)
            chunk.to_csv(f"chunk_part{i}.csv", index=True)

    df_dff, df_z = get_traces(file_traces)
    #main_df = get_traces_events(file_events, df_z)
    #split_by_gap(main_df)
    return df_z

main_df = split_traces_events(file_traces, file_events)


#%%

def get_behav(file_DLC, file_TTL, main_df, file_video, mirrorY=False):
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

            high_times = list(zip(onsets, offsets)) if not constant else list(onsets)
            print(f'{len(high_times)} TTLs')

            return high_times
        
        def get_video_dims(file_video):
            cap = cv2.VideoCapture(file_video)
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            print(f'{n_frames} frames')
            return (width, height)
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

            # adapt index
            df.set_index('bodyparts_coords', inplace=True)
            df.index.names = ['Frame']

            # turn all the prior string values into floats and index into int
            df = df.astype(float)
            df.index = df.index.astype(int)

            # insert NaN if likelihood is below x
            for bp in bps_all:
                filter = df[f'{bp}_likelihood'] <= 0.9     
                df.loc[filter, f'{bp}_x'] = np.nan
                df.loc[filter, f'{bp}_y'] = np.nan
                df = df.drop(labels=f'{bp}_likelihood', axis="columns")
            
            # mean 'trustworthy' bodyparts (ears, ear tips, eyes, midpoint, neck) to 'head' column
            for c in ('_x', '_y'):
                df[f'head{c}'] = df[[bp+c for bp in bps_head]].mean(axis=1, skipna=True)
                df[f'postbody{c}'] = df[[bp+c for bp in bps_postbody]].mean(axis=1, skipna=True)

            width, height = get_video_dims(file_video)
            if mirrorY:
                # mirror y and forget everything except head and postbody
                for bp in ['head', 'postbody']:
                    df[f'{bp}_y'] = height - df[f'{bp}_y']
            df_behav = df[['head_x', 'head_y', 'postbody_x', 'postbody_y']].copy()

            return df_behav
        
        df_behav = clean_behav(file_DLC)

        # as an index for behavior, you have the frame number and the INSPX master time
        TTLs_high_times = clean_TTLs(file_TTL)
        
        df_behav['Time'] = TTLs_high_times[:-1]
        df_behav['Time'] = df_behav['Time'].round(2)
        df_behav.set_index([df_behav.index, 'Time'], inplace=True)

        # interpolate the NaNs (according to past and future values)
        for col in ['head_x', 'head_y', 'postbody_x', 'postbody_y']:
            df_behav[col] = df_behav[col].interpolate(method='linear', limit_area='inside')
        df_behav = df_behav.round(1)

        # get start and end times of behav to trim the main_df after merging
        behav_start = round(df_behav.index.get_level_values('Time').min(), 1)
        behav_end = round(df_behav.index.get_level_values('Time').max(), 1)

        # makes index a regular column (necessary for merging) and merge with main_df
        main_df = main_df.reset_index()
        df_behav = df_behav.reset_index()
        main_df = pd.merge_asof(main_df, df_behav, on='Time', direction='nearest')
        main_df = main_df.set_index('Time')

        # index is regular time, that is limited by [first_tracked_Bp : video_max] 
        main_df = main_df.loc[behav_start : behav_end]
        main_df.index = (main_df.index - behav_start).round(1)

        return main_df

main_df = pd.read_csv(file_tracesevents, index_col='Time')
main_df = get_behav(file_DLC, file_TTL, main_df, file_video)
main_df.to_csv('main.csv')




#%%


#main_df = pd.read_csv(file_tracesevents, index_col='Time')
index1 = 0
index2 = 600

df_window = main_df.iloc[index1:index2]
mean_zscores = df_window.mean(axis=0)
sorted_cells = mean_zscores.sort_values(ascending=False).index
df_sorted = main_df[sorted_cells]





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

activity_heatmap(df_sorted)