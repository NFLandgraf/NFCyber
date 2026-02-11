#%%
'''
You need a csv file (from Excel file) with the cropping parameters for each video
Checks the csv file and according to this, changes the dimensions of the mp4 in a folder if name in there
Keeps everything else the same
'''
import os
import cv2
import pandas as pd

input_folder = r"D:\new\Behav_1_Raw\YM"
output_folder = r"D:\new\Behav_2_Crop"

# CSV with columns: filename, x1_new, x2_new, y1_new, y2_new, Here, "filename" is treated as the KEY to match as a substring in the .mp4 filenames
crop_data = pd.read_csv(r"D:\Video_Dimensions.csv")
crop_dict = {}
for _, row in crop_data.iterrows():
    key = str(row["filename"])
    crop_dict[key] = {
        "x1": int(row["x1_new"]),
        "x2": int(row["x2_new"]),
        "y1": int(row["y1_new"]),
        "y2": int(row["y2_new"])}


mp4_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".mp4")]
# Go through each thing in the dictionary and find hits in the folder
for key, crop_params in crop_dict.items():
    hits = [f for f in mp4_files if key in f]

    if not hits:
        print(f'\nNo videos found containing key "{key}"')
        continue

    x1, x2, y1, y2 = crop_params["x1"], crop_params["x2"], crop_params["y1"], crop_params["y2"]
    print(f'\n{key}')
    print(f'Hits: {hits}')

    # Crop all videos that match this key
    for filename in hits:
        video_path = os.path.join(input_folder, filename)
        base, ext = os.path.splitext(filename)
        output_video_path = os.path.join(output_folder, f"{base}_crop{ext}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  ❌ Could not open: {filename}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_w, out_h = (x2 - x1), (y2 - y1)

        # Create VideoWriter
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (out_w, out_h))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cropped = frame[y1:y2, x1:x2]
            out.write(cropped)

        cap.release()
        out.release()
        print(f"  ✅ Saved: {output_video_path}")

