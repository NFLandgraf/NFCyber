#%%
# IMPORT

import cv2
import ffmpeg
from tqdm import tqdm
import os

path = 'C:\\Users\\landgrafn\\NFCyber\\VideoAnalysis\\data\\'
path = 'C:\\Users\\landgrafn\\Desktop\\Paule\\3m\\'
common_name = '3m'
file_format = '.mp4'

# get all file names in directory into list
files = [file for file in os.listdir(path) 
             if os.path.isfile(os.path.join(path, file)) and
             common_name in file]
print(f'{len(files)} files found in path directory {path}\n'
      f'{files}\n')

# get video properties of all videos in list
for file in files:
      vid = cv2.VideoCapture(path + file)

      width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
      height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
      nframes = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
      fps = int(vid.get(cv2.CAP_PROP_FPS))
      duration = nframes / fps

      print(f'{file}\n'
            f'dimensions: {width} x {height} px\n'
            f'nframes: {nframes}\n'
            f'fps: {fps}\n'
            f'duration: {duration} s\n')
      


#%%####################################################################################################################
# CHANGE VIDEO PARAMETERS

# CROP
x1, y1 = 100, 0   # top left corner of future cropped image
x2, y2 = 1570, 1216    # bottom right corner of future cropped image

# BRIGHTNESS & CONTRAST
alpha = 1.25   # brightness: 1.0-original, <1.0-darker, >1.0-brighter
beta = -50    # contrast: 0-unchanged, <0-lower contrast, >0-higher contrast

def adjust_video(input_file, output_file, new_width, new_height, fps):
    cap = cv2.VideoCapture(input_file)
    first_frame = input_file.replace(file_format, '') + '_firstFrame.png'

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_file, fourcc, fps, (new_width, new_height)) 

    for curr_frame in tqdm(range(nframes)):    # nframes or 1
        ret, frame = cap.read()

        if ret:            
            # CROP
            #frame = frame[y1:y2, x1:x2]

            # BRIGHTNESS & CONTRAST
            frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

            # RESIZE
            #frame = cv2.resize(frame, (new_width, new_height))  # must be the same dimensions as in video = cv2.VideoWriter() 

            video.write(frame)

            # when you had range(1) to only check the first frame, write first frame. Delete it if >0
            if curr_frame == 0:
                cv2.imwrite(first_frame, frame)
            elif curr_frame == 1:
                os.remove(first_frame)

        else:
            break

    cap.release()
    video.release()

    print(f'{input_file} done! input: {nframes} frames, output: {curr_frame+1} frames\n\n')

for file in files:
    vid = cv2.VideoCapture(path + file)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nframes = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))

    #new_width, new_height = x2-x1, y2-y1                        # USE THIS WHEN     CROP     & NOT REDUCE QUALITY
    #new_width, new_height = int((x2-x1)*2/3), int((y2-y1)*2/3)  # USE THIS WHEN     CROP     &     REDUCE QUALITY
    #new_width, new_height = int(width/2), int(height/2)         # USE THIS WHEN NOT CROP     &     REDUCE QUALITY
    new_width, new_height = width, height                       # USE THIS WHEN NOT CROP     & NOT REDUCE QUALITY


    input_file = path + file
    output_file = input_file.replace(file_format, '') + '_edit' + file_format

    adjust_video(input_file, output_file, new_width, new_height, fps)








#%%####################################################################################################################
# TRIM VIDEO

input_file = path + files[0]
output_file = input_file.replace(file_format, '') + '_trim' + file_format

# trim video according to frames
output = ffmpeg.output(ffmpeg.input(input_file).trim(start_frame=0, end_frame=1000), output_file)
ffmpeg.run(output)