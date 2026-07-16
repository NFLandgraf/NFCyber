#%%
import pandas as pd
import numpy as np
from pathlib import Path
import os

'''
This script takes the sLEAP csv files (already Resi and Intr clearly stated) and does Quality control:
    1. removes tracks that are not Resi or Intr, e.g. track_1
    2. recovers missing frames, that sLEAP did not export due to missing animals
    3. checks when Intruder appeared for first and last time and corrects that
    4. converts this sLEAP-style into DLC-style
    5. does a sanity check if sLEAP-style and DLC-style are indeed the same values overall
'''

path = r"E:\AGG-FF\Videos_Raw_fps_sLEAP"

def get_files(path, common_name):

    # get files
    files = [file for file in Path(path).iterdir() if file.is_file() and common_name in file.name]
    
    # print(f'\n{len(files)} files found')
    # for file in files:
    #     print(file)
    # print('\n')

    return files

def check_tracks(df, extra_track='track_1'):

    # check wether there are more than 2 values in track (should only be Resi and Intr), change extra_track to Resi
    df_replaced = df.copy()
    df_replaced["track"] = df_replaced["track"].replace(extra_track, "Resi")
    if len(df_replaced["track"].unique()) > 2:
        print('Alert, more than Resi and Intr left')

    # now some of them could be Resi twice for the same frame_idx, delete the one with the lower score
    df_removed = df_replaced.copy()
    to_remove = (df_removed.sort_values("instance.score").duplicated(subset=["frame_idx", "track"], keep="last"))
    df_removed = df_removed.loc[~to_remove]

    if True:
        print('\ncheck tracks')
        print(  f"Resi: {(df['track'] == 'Resi').sum()}, "
                f"Intr: {(df['track'] == 'Intr').sum()}, "
                f"{extra_track}: {(df['track'] == extra_track).sum()}, "
                f"frames: {df['frame_idx'].nunique()}, "
                f"len: {len(df)}")
        print(  f"Resi: {(df_replaced['track'] == 'Resi').sum()}, "
                f"Intr: {(df_replaced['track'] == 'Intr').sum()}, "
                f"{extra_track}: {(df_replaced['track'] == extra_track).sum()}, "
                f"frames: {df_replaced['frame_idx'].nunique()}, "
                f"len: {len(df_replaced)}")
        print(  f"Resi: {(df_removed['track'] == 'Resi').sum()}, "
                f"Intr: {(df_removed['track'] == 'Intr').sum()}, "
                f"{extra_track}: {(df_removed['track'] == extra_track).sum()}, "
                f"frames: {df_removed['frame_idx'].nunique()}, "
                f"len: {len(df_removed)}")
        
    return df_removed

def recover_missing_frames(df, score_value=np.nan, min_frame=0, max_frame=35999, frame_col='frame_idx', track_col='track'):

    # identifies missing frame_idx and adds row for that by copying previous frames, adding np.nan for bodyparts and low score

    # get the data from the df
    coord_cols = [c for c in df.columns if c.endswith(".x") or c.endswith(".y")]
    score_cols = list(dict.fromkeys([c for c in df.columns if c.endswith(".score")] + ["instance.score"]))
    min_frame = min_frame if min_frame is not None else int(df[frame_col].min())
    max_frame = max_frame if max_frame is not None else int(df[frame_col].max())

    # finds missing frame_idx between min_frame and max_frame
    existing_frames = set(df[frame_col].dropna().astype(int))
    all_frames = set(range(min_frame, max_frame + 1))
    missing_frames = sorted(all_frames - existing_frames)

    new_rows = []
    for frame in missing_frames:

        
        base_row = df.iloc[0].copy()
        base_row[track_col] = "Resi"
        base_row[frame_col] = frame

        # adds np.nan for all x and y bodyparts
        for col in coord_cols:
            base_row[col] = np.nan

        # adds the low score_value 
        for col in score_cols:
            if col in base_row.index:
                base_row[col] = score_value

        new_rows.append(base_row)

    if new_rows:
        df_out = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    else:
        df_out = df.copy()

    df_out = df_out[df.columns]
    df_out = df_out.sort_values([frame_col, track_col]).reset_index(drop=True)

    if True:
        print('\nfill missing')
        print("Missing frames:", len(missing_frames))
        print(  f"Resi: {(df['track'] == 'Resi').sum()}, "
                f"Intr: {(df['track'] == 'Intr').sum()}, "
                f"Track_1: {(df['track'] == 'track_1').sum()}, "
                f"frames: {df['frame_idx'].nunique()}, "
                f"len: {len(df)}")
        print(  f"Resi: {(df_out['track'] == 'Resi').sum()}, "
                f"Intr: {(df_out['track'] == 'Intr').sum()}, "
                f"Track_1: {(df_out['track'] == 'track_1').sum()}, "
                f"frames: {df_out['frame_idx'].nunique()}, "
                f"len: {len(df_out)}")
        
    return df_out

def check_Intr_occurence(df, min_block=60, valid_Intr_range=(8100, 27900), track_col = 'track', frame_col = 'frame_idx', score_col = 'instance.score'):

    # deletes Intr rows before and after a prober first/last block
    # we take Intr blocks where their block is min_block frames long AND their last frame is above a limit (e.g. no Intr block expected after 16min)
    
    first_valid_Intr_frame, last_valid_Intr_frame = valid_Intr_range

    # identify Intr frame blocks
    intr_frames = (df.loc[df[track_col] == "Intr", frame_col].dropna().astype(int).drop_duplicates().sort_values())
    blocks = intr_frames.groupby(intr_frames.diff().ne(1).cumsum())
    valid_blocks = [block for _, block in blocks if len(block) >= min_block and block.iloc[-1] <= last_valid_Intr_frame and block.iloc[0] >= first_valid_Intr_frame]
    
    first_Intr_frame = valid_blocks[0].iloc[0]
    last_Intr_frame = valid_blocks[-1].iloc[-1]

    df_out = df.copy()

    outside_intr = ((df_out[track_col] == "Intr") & ((df_out[frame_col] < first_Intr_frame) | (df_out[frame_col] > last_Intr_frame)))

    n_changed = outside_intr.sum()
    df_out.loc[outside_intr, track_col] = "Resi"
    df_out = df_out.sort_values(score_col, ascending=False)
    df_out = df_out.drop_duplicates(subset=[frame_col, track_col], keep="first")
    df_out = df_out.sort_values([frame_col, track_col]).reset_index(drop=True)
    intr_frames_after = (df_out.loc[df_out[track_col] == "Intr", frame_col].dropna().astype(int).drop_duplicates().sort_values())

    if True:
        print('\nIntr occurence')
        print(f"Changed Intr -> Resi outside valid range: {n_changed}")
        print(f"Intr frame First: {round(intr_frames.min()/30/60, 2)}, Last: {round(intr_frames.max()/30/60, 2)}")
        print(f"Intr block First: {round(first_Intr_frame/30/60, 2)}, Last: {round(last_Intr_frame/30/60, 2)}")
        print(f"Intr frame First: {round(intr_frames_after.min()/30/60, 2)}, Last: {round(intr_frames_after.max()/30/60, 2)}")

    return df
    
def convert_sleap_to_dlc_style(df, scorer="sLEAP", track_map={"Resi": "1", "Intr": "2"}):
    
    bodyparts = ["Ear_left", "Ear_right", "Nose", "Center", "Lat_left", "Lat_right", "Tail_base", "Tail_end"]
    coords = ["x", "y", "likelihood"]
    rows = []

    for frame, group in df.groupby("frame_idx", sort=True):
        row = {"frame": float(frame)}

        for track_name, animal_id in track_map.items():
            animal_rows = group[group["track"] == track_name]
            animal = animal_rows.iloc[0] if len(animal_rows) > 0 else None

            for bp in bodyparts:
                if animal is None:
                    row[f"{bp}_{animal_id}_x"] = np.nan
                    row[f"{bp}_{animal_id}_y"] = np.nan
                    row[f"{bp}_{animal_id}_likelihood"] = np.nan
                else:
                    row[f"{bp}_{animal_id}_x"] = animal.get(f"{bp}.x", np.nan)
                    row[f"{bp}_{animal_id}_y"] = animal.get(f"{bp}.y", np.nan)
                    row[f"{bp}_{animal_id}_likelihood"] = animal.get(f"{bp}.score", np.nan)

        rows.append(row)

    out = pd.DataFrame(rows)
    ordered_cols = ["frame"]
    for track_name, animal_id in track_map.items():
        for bp in bodyparts:
            for coord in coords:
                ordered_cols.append(f"{bp}_{animal_id}_{coord}")

    out = out[ordered_cols]
    header_1 = ["frame"]
    header_2 = ["frame"]
    header_3 = [""]

    for track_name, animal_id in track_map.items():
        for bp in bodyparts:
            for coord in coords:
                header_1.append(scorer)
                header_2.append(f"{bp}_{animal_id}")
                header_3.append(coord)

    out.columns = pd.MultiIndex.from_arrays([header_1, header_2, header_3])

    return out


# sanity checks
def compare_sLEAP_vs_DLC(sleap_file, dlc_file):

    # checks the whole df between sLEAP and DLC
    sleap = pd.read_csv(sleap_file)
    dlc = pd.read_csv(dlc_file, header=[0, 1, 2])

    dlc.columns = ["frame" if col[0] == "frame" else f"{col[1]}_{col[2]}"for col in dlc.columns]
    dlc = dlc.set_index("frame")
    track_map = {"Resi": "1", "Intr": "2"}
    bodyparts = ["Ear_left", "Ear_right", "Nose", "Center", "Lat_left", "Lat_right", "Tail_base", "Tail_end"]

    mismatches = []

    for _, row in sleap.iterrows():
        frame = row["frame_idx"]
        track = row["track"]
        animal_id = track_map[track]

        if frame not in dlc.index:
            mismatches.append((frame, track, "missing frame in DLC"))
            continue

        dlc_row = dlc.loc[frame]
        for bp in bodyparts:
            for coord in ["x", "y"]:
                sleap_val = row[f"{bp}.{coord}"]
                dlc_val = dlc_row[f"{bp}_{animal_id}_{coord}"]

                if not np.isclose(sleap_val, dlc_val, equal_nan=True):
                    mismatches.append((frame, track, bp, coord, sleap_val, dlc_val))

    if True:
        if len(mismatches) == 0:
            print("\nsLEAP vs DLC check: All toppi")
        else:
            print(f"Found {len(mismatches)} mismatches.")
            print("First mismatches:")
            for m in mismatches[:20]:
                print(m)

def check_DLC_frames(folder):

    files = [f for f in os.listdir(folder) if f.lower().endswith(".csv") and 'DLC' in f]

    for file in sorted(files):

        path = os.path.join(folder, file)
        df = pd.read_csv(path, header=[0, 1, 2])

        frame_col = df.columns[0]
        frames = pd.to_numeric(df[frame_col], errors="coerce").astype("Int64")

        n_unique = frames.nunique()
        min_frame = frames.min()
        max_frame = frames.max()
        duplicates = frames.duplicated().sum()

        expected = set(range(int(min_frame), int(max_frame) + 1))
        missing = sorted(expected - set(frames.dropna()))

        print(file)
        print(f"{min_frame} - {max_frame}: {n_unique} unique frames in {len(df)} rows")
        print(f"{duplicates} duplicates, {len(missing)} missing frames\n")


files = get_files(path, '')

for file_sLEAP in files:

    print(file_sLEAP)
    df = pd.read_csv(file_sLEAP)

    # do stuff
    df = check_tracks(df)
    df = recover_missing_frames(df)
    df_sleap = check_Intr_occurence(df)
    df_dlc   = convert_sleap_to_dlc_style(df_sleap)

    # save files
    fileout_sLEAP = os.path.splitext(file_sLEAP)[0] + "_check.csv"
    fileout_DLC = os.path.splitext(file_sLEAP)[0] + "_check_DLC.csv"
    df_sleap.to_csv(fileout_sLEAP, index=False)
    df_dlc.to_csv(fileout_DLC, index=False)

    # sanity checks
    compare_sLEAP_vs_DLC(fileout_sLEAP, fileout_DLC)

check_DLC_frames(path)
