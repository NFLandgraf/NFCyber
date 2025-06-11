import pandas as pd
import os
import matplotlib.pyplot as plt

# --------- SETTINGS ----------
path = 'D:\\'
file_format = '.csv'
bodyparts = ['Center', 'Nose', 'Tail_base', 'Tail_end', 'Ear_right', 'Ear_left', 'Lat_right', 'Lat_left']
animal_ids = {'Resi': '1', 'Intr': '2'}
scorer = 'sLEAP'
# -----------------------------

def get_data():
    files = [file for file in os.listdir(path) 
             if os.path.isfile(os.path.join(path, file)) and file_format in file]
    print(f'{len(files)} files found in path directory {path}\n{files}\n')
    return files

files = get_data()

def change(file):
    print(file)
    input_file = os.path.join(path, file)
    output_file = input_file.replace(file_format, '') + '_DLC' + file_format

    # Load original sLEAP CSV
    df = pd.read_csv(input_file)

    # Group by frame and reshape into one row per frame
    frames = df['frame_idx'].unique()
    data_rows = []
    for frame in frames:
        row = {'frame': frame}
        frame_data = df[df['frame_idx'] == frame]
        for _, animal in frame_data.iterrows():
            track = animal['track']
            animal_id = animal_ids.get(track.strip().capitalize(), None)
            if not animal_id:
                continue
            for bp in bodyparts:
                row[f'{bp}_{animal_id}_x'] = animal.get(f'{bp}.x', pd.NA)
                row[f'{bp}_{animal_id}_y'] = animal.get(f'{bp}.y', pd.NA)
                row[f'{bp}_{animal_id}_p'] = animal.get(f'{bp}.score', pd.NA)
        data_rows.append(row)

    reshaped_df = pd.DataFrame(data_rows)

    # Construct DLC-style header
    header_rows = [['frame'], [''], ['']]  # frame column headers
    for bp in bodyparts:
        for animal_id in ['1', '2']:
            header_rows[0] += [scorer, scorer, scorer]
            header_rows[1] += [f'{bp}_{animal_id}'] * 3
            header_rows[2] += ['x', 'y', 'likelihood']

    # Create MultiIndex DataFrame
    columns = pd.MultiIndex.from_arrays(header_rows, names=['scorer', 'bodyparts', 'coords'])
    dlc_df = pd.DataFrame(reshaped_df.values, columns=columns)

    # Save to CSV
    dlc_df.to_csv(output_file, index=False)
    print(f"✅ DLC-style file saved to: {output_file}")

for file in files:
    change(file)
