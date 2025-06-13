#%%
import subprocess
import os
import cv2
from moviepy.editor import ImageSequenceClip

path = 'D:\\Videos_to_10fps\\'
file_format = '.mp4'

def get_data():
    # get all file names in directory into list
    files = [file for file in os.listdir(path) 
                if os.path.isfile(os.path.join(path, file)) and
                file_format in file]
    print(f'{len(files)} files found in path directory {path}\n'
        f'{files}\n')
    
    return files
files = get_data()


#%%


def reencode_for_solomon(file):

    input_file = path + file
    output_file = input_file.replace(file_format, '') + '_solo' + file_format
    print(input_file)
    

    command = [
        "ffmpeg",
        "-i", input_file,
        "-c:v", "libx264",         # widely supported
        "-preset", "ultrafast",    # fast encoding
        "-crf", "18",              # high quality
        "-pix_fmt", "yuv420p",     # safe pixel format
        "-vsync", "vfr",           # keep original timing
        "-map", "0",               # include all streams
        "-y",                      # overwrite without asking
        output_file
    ]

    subprocess.run(command, check=True)
    print("✅ Done.")


for file in files:
    reencode_for_solomon(file)

    input_file = path + file
    cap = cv2.VideoCapture(input_file)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    print(n_frames)
    print(fps)
    print(width)
    print(height)
    print(codec)

    output_file = input_file.replace(file_format, '') + '_solo' + file_format
    cap = cv2.VideoCapture(output_file)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    print(n_frames)
    print(fps)
    print(width)
    print(height)
    print(codec)



#%%
# MAKE FRAMES

for file in files:
    input_file = path + file
    print(input_file)

    output_folder = f'{path}{file}_frames'
    os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(input_file)

    # Get the total number of frames and the original frame rate
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Extract each frame and save as an image
    for frame_idx in range(frame_count):
        ret, frame = cap.read()
        if ret:
            # Save each frame as an image (frame_0001.jpg, frame_0002.jpg, etc.)
            cv2.imwrite(f"{output_folder}\\{frame_idx:04d}.jpg", frame)

    # Release the video capture object
    cap.release()



#%%

# MAKE VIDEO FROM FRAMES

for file in files:
    input_file = path + file
    output_file = input_file.replace(file_format, '') + '_solo' + file_format
    print(input_file)

    # Folder with frames
    frame_folder = f"{path}\\{file}_frames"

    # Desired frame rate for the new video (e.g., 10 fps)
    new_fps = 10

    # Load all the frames from the folder
    frame_files = [f"{frame_folder}/{i:04d}.jpg" for i in range(frame_count)]
    print(len(frame_files))

    # Create a video from the frames
    clip = ImageSequenceClip(frame_files, fps=new_fps)

    # Write the new video to a file
    output_video = output_file
    clip.write_videofile(output_video, codec="libx264")


#%%

# Open the video file
cap = cv2.VideoCapture('D:\\a\\10.mp4')

# Get the total number of frames and the original frame rate
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(frame_count)