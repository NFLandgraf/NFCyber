#%%
'''
SimBA does not like our sLEAP-style csvs.
This script takes the raw csv from sLEAP and changes it into a DLC-style csv.
It also changes the column order so that SimBA uses the correct bodyparts.
'''
import pandas as pd
import os

path = r"C:\Users\landgrafn\Desktop\sLEAP_done"
file_format = ".csv"

# names as they exist in the original sLEAP csv
sleap_bodyparts = ['Center', 'Nose', 'Tail_base', 'Tail_end', 'Ear_right', 'Ear_left', 'Lat_right', 'Lat_left']

# order SimBA seems to expect
output_bodyparts = ['Ear_left', 'Ear_right', 'Nose', 'Center', 'Lat_left', 'Lat_right', 'Tail_base', 'Tail_end']

animal_ids = {'Resi': '1', 'Intr': '2'}
animal_id_again = ['1', '2']
scorer = 'sLEAP'

def change(file):
    print(f'Processing: {file}')
    input_file = os.path.join(path, file)
    output_file = input_file.replace(file_format, '') + '_DLC' + file_format
    df = pd.read_csv(input_file)

    print("Unique track names in file:", df['track'].dropna().unique())

    frames = sorted(df['frame_idx'].unique())
    data_rows = []

    for frame in frames:
        row = {'frame': frame}
        frame_data = df[df['frame_idx'] == frame]

        for _, animal in frame_data.iterrows():
            track = str(animal['track']).strip()
            animal_id = animal_ids.get(track, None)
            if animal_id is None:
                continue

            for bp in sleap_bodyparts:
                row[f'{bp}_{animal_id}_x'] = animal.get(f'{bp}.x', pd.NA)
                row[f'{bp}_{animal_id}_y'] = animal.get(f'{bp}.y', pd.NA)
                row[f'{bp}_{animal_id}_likelihood'] = animal.get(f'{bp}.score', pd.NA)

        data_rows.append(row)

    reshaped_df = pd.DataFrame(data_rows)

    ordered_cols = ['frame']
    for animal_id in animal_id_again:
        for bp in output_bodyparts:
            ordered_cols += [
                f'{bp}_{animal_id}_x',
                f'{bp}_{animal_id}_y',
                f'{bp}_{animal_id}_likelihood'
            ]

    reshaped_df = reshaped_df.reindex(columns=ordered_cols)

    top = ['frame']
    mid = ['frame']
    low = ['']

    for animal_id in animal_id_again:
        for bp in output_bodyparts:
            top += [scorer, scorer, scorer]
            mid += [f'{bp}_{animal_id}', f'{bp}_{animal_id}', f'{bp}_{animal_id}']
            low += ['x', 'y', 'likelihood']

    multi_cols = pd.MultiIndex.from_arrays([top, mid, low])
    dlc_df = pd.DataFrame(reshaped_df.values, columns=multi_cols)

    dlc_df.to_csv(output_file, index=False)
    print(f'✅ Saved: {output_file}')

files = [file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file)) and file.endswith(file_format)]
for file in files:
    change(file)

#%%
# Raw Solomon to one Behav
import pandas as pd
import os

input_folder = r"C:\Users\landgrafn\Desktop\FINAL\Own_Behav_Solomon"
output_folder = r"C:\Users\landgrafn\Desktop\FINAL\SimBA_TailRattle_import_Own_Behav_SolomonStyle"
behav = 'Tail_Rattle'

os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):
    if file.endswith(".csv"):
        path = os.path.join(input_folder, file)
        
        df = pd.read_csv(path)
        
        # keep only behav
        df["Behaviour"] = df["Behaviour"].apply(lambda x: behav if x == behav else "")
        
        out_path = os.path.join(output_folder, file)
        df.to_csv(out_path, index=False)