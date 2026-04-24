#%%
# you have a folder with OSF-annotated csv, check if they are all in the video_info csv
import os
import pandas as pd

folder_path = r"C:\Users\landgrafn\Desktop\FINAL\SimBA_OSF_Tail_Rattle_Raw"
csv_path = r"C:\Users\landgrafn\Desktop\FINAL\OSF_video_info.csv"
csv_column = "Video"
file_extension = ".csv"
behav = 'Tail_Rattle'

folder_files = [
    os.path.splitext(f)[0]
    for f in os.listdir(folder_path)
    if os.path.isfile(os.path.join(folder_path, f)) and f.endswith(file_extension)
]

print(f"Files in folder: {len(folder_files)}")

df = pd.read_csv(csv_path)
csv_files = df[csv_column].astype(str).apply(lambda x: os.path.splitext(x)[0]).tolist()

folder_set = set(folder_files)
csv_set = set(csv_files)

missing_in_csv = folder_set - csv_set
extra_in_csv = csv_set - folder_set

print(f"\nFiles in folder but NOT in CSV ({len(missing_in_csv)}):")
for f in sorted(missing_in_csv):
    print(f)
print(f"Entries in CSV but NOT in folder ({len(extra_in_csv)}):")
if len(missing_in_csv) == 0:
    print("✅ All folder files are present in the CSV")
else:
    print("❌ Some files are missing in the CSV")

# check that the behav is found in the csvs
files_missing_behav = []
for file in os.listdir(folder_path):
    if file.endswith(".csv"):
        file_path = os.path.join(folder_path, file)
        try:
            df_file = pd.read_csv(file_path, nrows=1)  # only header needed
            if behav not in df_file.columns:
                files_missing_behav.append(file)
        except Exception as e:
            print(f"⚠️ Error reading {file}: {e}")

print(f"Files missing {behav} column ({len(files_missing_behav)}):")
if len(files_missing_behav) == 0:
    print(f"✅ All CSV files contain {behav}")
else:
    print(f"❌ Some files are missing {behav}")

#%%
# use this to change a specific string in the header of the OSF csv, e.g. if a behavior is written differently
import os
import pandas as pd


folder_path = r"Y:\Neuronal Networks\SimBA\pretrained_Mouse_RI\Tail_rattle"
old_string = "Tail_rattle"
new_string = "Tail_Rattle"

files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(".csv")]

for file in files:
    file_path = os.path.join(folder_path, file)
    print(f'Processing: {file}')

    df = pd.read_csv(file_path)
    new_columns = [col.replace(old_string, new_string) for col in df.columns]
    df.columns = new_columns
    df.to_csv(file_path, index=False)

    print("    Header updated")

print("Done.")