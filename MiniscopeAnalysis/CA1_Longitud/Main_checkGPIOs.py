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
from scipy.ndimage import gaussian_filter1d
import tifffile as tiff


def get_files(path, common_name, suffix=".csv"):
    path = Path(path)
    files = sorted([p for p in path.iterdir() if p.is_file() and (common_name in p.name) and p.name.endswith(suffix)])
    print(f"\n{len(files)} files found")
    for f in files:
        print(f)
    print('\n\n')
    return files
path = r"D:\test"
files = get_files(path, 'gpio', suffix=".csv")


def fuse(file_DLC, file_TTL, file_video, file_motcor, file_master_out, mirrorY=False):
        
        # this takes the file_DLC and file_TTL and checks wether they have the same numbers, then put the file_TTL time on DLC
        # this takes the file_motcor and file_TTL and checks wether they have the same number, then takes the file_TTL Neuro time
        # afterwards, it puts the file_TTL Neuro time on the df_behav, so we have one big df_master

        # cleans and merges behav stuff
        def clean_TTLs(file_TTL, cam_name=' GPIO-1', cam_threshold=10000, constant=True):
            # returns list of TTL input times

            # expects nan values to be just a string: ' nan', convert these strings into proper nans
            df = pd.read_csv(file_TTL, na_values=[' nan'], low_memory=False)
            df = df.set_index('Time (s)')

            # only cares for the rows where the channel name is in the ' Channel Name' column
            df = df[df[' Channel Name'] == cam_name]
            
            # drops rows with nan and converts everything into ints
            df = df.dropna(subset=[' Value'])
            df[' Value'] = df[' Value'].astype(int)

            # creates series with True for values > threshold
            # checks at which index/time the row's value is high AND the row before's value is low
            # for that, it makes the whole series negative (True=False) and shifts the whole series one row further (.shift)
            # so, onset wherever row is > threshold and the opposite value of the shifted (1) row is positive as well
            is_high = df[' Value'] >= cam_threshold
            onsets = df.index[is_high & ~is_high.shift(1, fill_value=False)]
            offsets = df.index[~is_high & is_high.shift(1, fill_value=False)]

            # high times are all the times where signal goes from below to above threshold
            high_times = list(zip(onsets, offsets)) if not constant else list(onsets)
            print(f'{len(high_times)} Cam_ttl')

            # sanity check
            high_times_arr = np.array(high_times)
            high_times_intervals = np.round(np.diff(high_times_arr), 5)
            unique_vals, counts = np.unique(high_times_intervals, return_counts=True)
            if unique_vals > 1:
                print('\nCam_TTL time intervals:')
                print(f'unique_valus: {unique_vals}')
                print(f'unique_count: {counts}')

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
            if 'OF' in file_DLC:
                scale_x = 460 / width   # mm/px
                scale_y = 460 / height  # mm/px
                #print('OF! for transformation px into mm ')
            elif 'YM' in file_DLC:
                scale_x = 580 / width   # mm/px
                scale_y = 485 / height  # mm/px
                #print('YMaze! for transformation px into mm ')
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

            # smoothing along time via gaussian filter
            for col in df.columns:
                df[col] = gaussian_filter1d(df[col].values, sigma=2, mode="nearest")

            print(f'{df.shape[0]} Cam_frames')
            return df
        
        def fuse_behav_TTL(df_behav, TTLs_high_times):

            Cam_frames = len(df_behav)
            Cam_TTLs = len(TTLs_high_times)

            if Cam_TTLs != Cam_frames:

                if Cam_TTLs > Cam_frames:
                    df_missing = pd.DataFrame(np.nan,index=range(Cam_frames, Cam_TTLs),columns=df_behav.columns)
                    df_behav = pd.concat([df_behav, df_missing])

                    # sanity check
                    n_after = len(df_behav)
                    added = n_after - Cam_frames
                    print(f'WARNING: Cam_ttl ({Cam_TTLs}) and Cam_frames ({Cam_frames}) -> {added} Cam_frames rows added (now {n_after})')

                else:
                    raise ValueError('more DLC frames than TTLs')
            
            if len(df_behav) != len(TTLs_high_times):
                raise ValueError('Immernoch Cam_frames != Cam_TTLs')
            
            # change index
            df_behav["Cam_frames"] = df_behav.index.astype(int)
            #df_behav["Cam_ttl"] = TTLs_high_times
            df_behav.index = TTLs_high_times
            df_behav.index.name = 'Mastertime [s]'
            
            return df_behav


        # cleans and merges Neuro stuff
        def get_frametimes(file_TTL, neur_name=' BNC Sync Output', neur_threshold=0.9, constant=True):

            # get everything in the correct order
            df = pd.read_csv(file_TTL, na_values=[' nan'], low_memory=False)
            df = df.set_index('Time (s)')
            df = df[df[' Channel Name'] == neur_name]
            df = df.dropna(subset=[' Value'])
            df[' Value'] = df[' Value'].astype(int)


            is_high = df[' Value'] == 1
            onsets = df.index[is_high & ~is_high.shift(1, fill_value=False)]
            timepoints_neur = np.array(list(onsets))

            # sanity check
            timepoints_neur_intervals = np.round(np.diff(timepoints_neur), 3)
            unique_vals, counts = np.unique(timepoints_neur_intervals, return_counts=True)
            if unique_vals > 1:
                print(f'\nNeuro_ttl: unique_valus: {unique_vals}')
                print(f'nNeuro_ttl: unique_count: {counts}')
            #print(f'{len(timepoints_neur)} Neuro_ttl')

            # downsample the list to 10Hz (mean 2 timestamps, drop the last in odd)
            timepoints_neur_odd_dropped = timepoints_neur[:len(timepoints_neur) - len(timepoints_neur) % 2]
            timepoints_neur_downsampled = timepoints_neur_odd_dropped.reshape(-1, 2).mean(axis=1)
            print(f'\n{len(timepoints_neur_downsampled)} Neuro_ttl downsampled')

            # create df for later
            frames_neur_downsampled = np.arange(len(timepoints_neur_downsampled))
            df_10Hz = pd.DataFrame(data={'Neuro_frames_10Hz': frames_neur_downsampled}, index=pd.Index(timepoints_neur_downsampled, name='Mastertime [s]'))

            return df_10Hz

        def check_motcor_frames(file_motcor, df_neur_10Hz):

            with tiff.TiffFile(file_motcor) as tif:
                Neuro_frames = len(tif.pages)
            print(f"{Neuro_frames} Neuro_frames")

            if Neuro_frames != len(df_neur_10Hz):
                raise ValueError('Neuro_frames != Neuro_ttl')


        # merges behav and neuro into df_master
        def fuse_neur_behavTTL(df_neur_10Hz, df_behav_TTL):
            
            # merges Neuro and Cam
            df_master = pd.merge_asof(
                            df_neur_10Hz.sort_index().reset_index().rename(columns={"Mastertime [s]": "df_neuro"}),
                            df_behav_TTL.sort_index().reset_index().rename(columns={"Mastertime [s]": "Cam_ttl"}),
                            left_on="df_neuro", right_on="Cam_ttl", direction="nearest")
            df_master = df_master.set_index("df_neuro")
            df_master.index.name = 'Mastertime [s]'

            # because the Neuro index is bigger, this clips the repeating start and end of Cam rows
            columns_df2 = [c for c in df_master.columns if c not in df_neur_10Hz.columns]
            Cam_frames_min = df_master["Cam_frames"].min()
            Cam_frames_max = df_master["Cam_frames"].max()
            mask_edge = df_master["Cam_frames"].isin([Cam_frames_min, Cam_frames_max])
            df_master.loc[mask_edge, columns_df2] = np.nan

            df_master['Cam_frames'] = df_master['Cam_frames'].astype("Int64")

            return df_master




        df_behav = clean_behav(file_DLC, file_video)
        TTLs_high_times = clean_TTLs(file_TTL)
        df_behav_TTL = fuse_behav_TTL(df_behav, TTLs_high_times)

        df_neur_10Hz = get_frametimes(file_TTL)
        check_motcor_frames(file_motcor, df_neur_10Hz)

        df_master = fuse_neur_behavTTL(df_neur_10Hz, df_behav_TTL)


        df_master.to_csv(file_master_out)
        return df_master

def get_trimming_file(filebase, file_trimming, start_extraframes=120):

    df_trim = pd.read_csv(file_trimming)

    match = df_trim[df_trim["filename"].str.contains(filebase, regex=False)]

    if len(match) == 1:
        behav_start = int(match["behav_start"].iloc[0]) + start_extraframes
        behav_end   = match["behav_end"].iloc[0]
    else:
        raise ValueError(f"Expected 1 match for {filebase}, found {len(match)}")
    
    if behav_end == 'ind':
        return (behav_start, behav_end)
    else:
        return (behav_start, int(behav_end))

def trim(df_master, file_video, file_motcor, behav_borders, file_master_trim_out, file_video_trim_out, file_motcor_trim_out):
    
    def trim_master(start_mastertime, end_mastertime, file_master_trim_out):

        df_master_trim = df_master.loc[start_mastertime:end_mastertime].copy()
        frames_keeping_for_trimming_mp4 = df_master_trim["Cam_frames"].tolist()

        # re-zero columns
        df_master_trim["Neuro_frames_10Hz"] = (df_master_trim["Neuro_frames_10Hz"] - df_master_trim["Neuro_frames_10Hz"].iloc[0])
        df_master_trim["Cam_ttl"] = (df_master_trim["Cam_ttl"] - df_master_trim["Cam_ttl"].iloc[0])
        df_master_trim["Cam_frames"] = (df_master_trim["Cam_frames"] - df_master_trim["Cam_frames"].iloc[0])
        df_master_trim.index = (df_master_trim.index - df_master_trim.index[0]).round(3)

        out_frames_Master = len(df_master_trim)

        df_master_trim.to_csv(file_master_trim_out)
        return out_frames_Master, frames_keeping_for_trimming_mp4

    def trim_mp4(file_video, keep_frames, file_video_trim_out):

        idx_set = set(int(i) for i in keep_frames if i is not None)
        if not idx_set:
            raise ValueError("frame_indices is empty")

        cap = cv2.VideoCapture(file_video)
        video_input_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = 10.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(file_video_trim_out, fourcc, fps, (width, height))

        frame_i, out_frames_Cam = 0, 0
        with tqdm(total=video_input_frames, desc="trim_mp4") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_i in idx_set:
                    out.write(frame)
                    out_frames_Cam += 1
                frame_i += 1
                pbar.update(1)

        cap.release()
        out.release()

        return out_frames_Cam

    def trim_tif(file_motcor, file_motcor_trim_out):

        print('trim_TIF')
        start, end = start_Neuro_frames, end_Neuro_frames

        with tiff.TiffFile(file_motcor) as tif:
            n = len(tif.pages)
            if end >= n or start < 0:
                raise ValueError('Invalid start/end')
            frames = tif.asarray(key=slice(start, end + 1))
        tiff.imwrite(file_motcor_trim_out, frames)

        return len(frames)


    start_Cam_frames, end_Cam_frames = behav_borders
    start_mastertime = (df_master["Cam_frames"] - start_Cam_frames).abs().idxmin()
    start_Neuro_frames = df_master.loc[start_mastertime, "Neuro_frames_10Hz"]
    end_Cam_frames = df_master["Cam_frames"].max() if end_Cam_frames == 'ind' else end_Cam_frames
    end_mastertime = (df_master["Cam_frames"] - end_Cam_frames).abs().idxmin()
    end_Neuro_frames = df_master.loc[end_mastertime, "Neuro_frames_10Hz"]

    print(f'\n{round(start_mastertime, 3)} - {round(end_mastertime, 3)} (Start-End Mastertime)')

    out_frames_Master, frames_keeping_for_trimming_mp4 = trim_master(start_mastertime, end_mastertime, file_master_trim_out)
    out_frames_Cam = trim_mp4(file_video, frames_keeping_for_trimming_mp4, file_video_trim_out)
    out_frames_Neuro = trim_tif(file_motcor, file_motcor_trim_out)

    if out_frames_Master == out_frames_Cam == out_frames_Neuro:
        print(f'{out_frames_Master} = {out_frames_Cam} = {out_frames_Neuro} (Master-Cam-Neuro frames)')
    else:
        raise ValueError('trimming went wrong')



file_trimming = "D:\\1stuff\\Video_TrimFrames.csv"
for gpio_file in files:

    # create base
    name = gpio_file.name
    base = name[:-len("_gpio.csv")]
    print(str(gpio_file.with_name(base)))


    # define all files
    file_TTL    = gpio_file
    file_DLC    = gpio_file.with_name(f"{base}_crop_mask_DLC.csv")
    file_video  = gpio_file.with_name(f"{base}_crop_mask.mp4")
    file_motcor = gpio_file.with_name(f"{base}_motcor.tif")

    file_master_out = gpio_file.with_name(f"{base}_master.csv")
    file_master_trim_out = gpio_file.with_name(f"{base}_master_trim.csv")
    file_video_trim_out  = gpio_file.with_name(f"{base}_crop_mask_trim.mp4")
    file_motcor_trim_out = gpio_file.with_name(f"{base}_motcor_trim.tif")


    # if a file is missing
    missing = [p for p in [file_DLC, file_video, file_motcor] if not p.exists()]
    if missing:
        print("Missing files:")
        for m in missing:
            print("  -", m)
        continue


    # main actions
    df_master = fuse(str(file_DLC), str(file_TTL), str(file_video), str(file_motcor), file_master_out)
    behav_borders = get_trimming_file(base, file_trimming)
    trim(df_master, str(file_video), str(file_motcor), behav_borders, file_master_trim_out, file_video_trim_out, file_motcor_trim_out)




#%%
