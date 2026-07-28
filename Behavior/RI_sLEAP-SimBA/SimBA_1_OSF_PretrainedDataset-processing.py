#%%
# Takes Raw_OSFfile and only keep the pose estimation columns
import pandas as pd
import os

input_folder = r"C:\Users\landgrafn\Desktop\SIMBA\SimBA_Model_TailRattle\SimBA_TailRattle_OSF_Raw"
output_folder = r"C:\Users\landgrafn\Desktop\SIMBA\SimBA_Model_TailRattle\new"

os.makedirs(output_folder, exist_ok=True)

pose_cols = [
    "Ear_left_1_x","Ear_left_1_y","Ear_left_1_p",
    "Ear_right_1_x","Ear_right_1_y","Ear_right_1_p",
    "Nose_1_x","Nose_1_y","Nose_1_p",
    "Center_1_x","Center_1_y","Center_1_p",
    "Lat_left_1_x","Lat_left_1_y","Lat_left_1_p",
    "Lat_right_1_x","Lat_right_1_y","Lat_right_1_p",
    "Tail_base_1_x","Tail_base_1_y","Tail_base_1_p",
    "Tail_end_1_x","Tail_end_1_y","Tail_end_1_p",
    "Ear_left_2_x","Ear_left_2_y","Ear_left_2_p",
    "Ear_right_2_x","Ear_right_2_y","Ear_right_2_p",
    "Nose_2_x","Nose_2_y","Nose_2_p",
    "Center_2_x","Center_2_y","Center_2_p",
    "Lat_left_2_x","Lat_left_2_y","Lat_left_2_p",
    "Lat_right_2_x","Lat_right_2_y","Lat_right_2_p",
    "Tail_base_2_x","Tail_base_2_y","Tail_base_2_p",
    "Tail_end_2_x","Tail_end_2_y","Tail_end_2_p"
]

for file in os.listdir(input_folder):
    if file.endswith(".csv"):
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, file)

        try:
            df = pd.read_csv(input_path)

            cols = [df.columns[0]] + [c for c in pose_cols if c in df.columns]
            df = df[cols]

            df.to_csv(output_path, index=False)
            print(f"Processed: {file}")

        except Exception as e:
            print(f"Error with {file}: {e}")

#%%
# Takes Raw_OSFfile and transforms the respective behav into SOlomonSytle
import pandas as pd
import os

fps = 30
behav = "Tail_Rattle"

input_folder = r"C:\Users\landgrafn\Desktop\FINAL\SimBA_TailRattle_OSF_Raw"
output_folder = r"C:\Users\landgrafn\Desktop\FINAL\SimBA_TailRattle_import_OSF_Behav_SolomonStyle"

os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):
    if file.endswith(".csv"):
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, file)

        try:
            df = pd.read_csv(input_path)

            first_col = df.columns[0]

            if behav not in df.columns:
                print(f"Skipped {file}: column '{behav}' not found")
                continue

            df = df[[first_col, behav]].copy()
            df["Time"] = (df[first_col] / fps).round(2)
            df["Behaviour"] = df[behav].replace({0: "", 1: behav})
            df = df[["Time", "Behaviour"]]

            df.to_csv(output_path, index=False)
            print(f"Processed: {file}")

        except Exception as e:
            print(f"Error with {file}: {e}")