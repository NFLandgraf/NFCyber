#%%
# goes through videos in folder and saves the first frame
import os
import cv2

# Path to the folder containing the .mp4 videos
input_folder = "D:\\CA1Dopa_Longitud_Videos\\all videos"

# Create an output folder if it doesn't exist
output_folder = "D:\\CA1Dopa_Longitud_Videos\\all videos firstframe"

# Loop through all files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.mp4'):
        print(filename)
        video_path = os.path.join(input_folder, filename)
        
        
        
        # Open the video
        cap = cv2.VideoCapture(video_path)
        
        # Read the first frame
        ret, frame = cap.read()
        
        if ret:
            # Save the first frame as a PNG
            frame_filename = os.path.join(video_path, video_path + '.png')
            cv2.imwrite(frame_filename, frame)
        
        # Release the video capture object
        cap.release()

print("First frames extracted and saved successfully!")
