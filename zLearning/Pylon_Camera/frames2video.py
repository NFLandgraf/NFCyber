import cv2
import os

parent_folder = 'C:\\Users\\Admin\\Nico_SSD_Cam\\'
frame_folder = 'Testy'

output_video = f'{parent_folder + frame_folder}\\{frame_folder}.mp4'
frame_folder = parent_folder + frame_folder
frame_rate = 30

# Get a list of all JPG files in the folder, sorted numerically
frame_files = sorted(
    [f for f in os.listdir(frame_folder) if f.endswith('.jpg')],
    key=lambda x: int(x.split('_')[1].split('.')[0]))

# Read the first frame to get the frame size (width, height)
first_frame = cv2.imread(os.path.join(frame_folder, frame_files[0]))
height, width, _ = first_frame.shape

# Initialize the VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format ('XVID' for AVI)
out = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

# Loop through all frames and write them to the video
for frame_file in frame_files:
    frame_path = os.path.join(frame_folder, frame_file)
    frame = cv2.imread(frame_path)

    if frame is not None:
        out.write(frame)
        print(f"{frame_file} to video")
    else:
        print(f"Failed to read {frame_file}.")

# Release the VideoWriter object and clean up
out.release()
cv2.destroyAllWindows()

print(f"Video saved as {output_video}.")
