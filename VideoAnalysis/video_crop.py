import os
import cv2
import pandas as pd

# Path to the folder containing the .mp4 videos
input_folder = "C:\\Users\\landgrafn\\Desktop\\CA1Dopa_Longitud_Videos"

# Output folder for the cropped videos
output_folder = "C:\\Users\\landgrafn\\Desktop\\CA1Dopa_Longitud_Videos\\crop_new"
os.makedirs(output_folder, exist_ok=True)

crop_data = pd.read_csv("C:\\Users\\landgrafn\\Desktop\\Croppo_new.CSV")

# Create a dictionary to store cropping coordinates by filename
crop_dict = {}
for index, row in crop_data.iterrows():
    filename = os.path.basename(row['filename'])  # Extract just the file name from the full path
    crop_dict[filename] = {
        'x1_new': row['x1_new'],
        'x2_new': row['x2_new'],
        'y1_new': row['y1_new'],
        'y2_new': row['y2_new']
    }


# Loop through all files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.mp4'):
        video_path = os.path.join(input_folder, filename)
        filename_new = filename.strip('.mp4')
        print(filename)

        if filename in crop_dict:
            crop_params = crop_dict[filename]
            x1, x2, y1, y2 = crop_params['x1_new'], crop_params['x2_new'], crop_params['y1_new'], crop_params['y2_new']
        
        # Create an output file path
        output_video_path = os.path.join(output_folder, f'{filename_new}_crop.mp4')
        
        # Open the video
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec if necessary
        
        # Create VideoWriter to save the cropped video
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (x2 - x1, y2 - y1))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Crop the frame using the given coordinates
            cropped_frame = frame[y1:y2, x1:x2]
            
            # Write the cropped frame to the output video
            out.write(cropped_frame)
        
        # Release video objects
        cap.release()
        out.release()
        print('video done')

print("Videos cropped and saved successfully!")
