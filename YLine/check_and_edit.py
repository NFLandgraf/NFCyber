#%%
# IMPORT
import cv2
from matplotlib import pyplot as plt 
from matplotlib import image 
from shapely.geometry import Polygon, Point
import os
from tqdm import tqdm
import numpy as np

def get_files(path):
    # get all file names in directory into list
    files = [file for file in os.listdir(path) 
                if os.path.isfile(os.path.join(path, file)) and
                'mp4' in file]
    
    return files

def get_xmax_ymax(file):
    # get video properties
    vid = cv2.VideoCapture(file)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    return width, height

path = 'C:\\Users\\landgrafn\\NFCyber\\YLine\\data\\'
file_format = '.mp4'

files = get_files(path)
width, height = get_xmax_ymax(path+files[0])

def adapt(x, y):
    # add dx/dy and multiplay with dz
    adapt_coords = ((x + dx) * dz, (y + dy) * dz)

    # check if poits are out of the image
    if 0 > adapt_coords[0] or adapt_coords[0] >= width:
        print('point exceeds width')
    elif 0 > adapt_coords[1] or adapt_coords[1] >= width:
        print('point exceeds height')

    return adapt_coords


# USER INPUT
dx = 0      # if you want to shift the whole thing horizontally 
dy = 0      # if you want to shift the whole thing vertically
dz = 1      # if you want to change the size of the whole thing

# corners
left_corner = adapt(350, 250)
middle_corner = adapt(390, 180)
right_corner = adapt(430, 250)

# left arm
left_arm_end_lefter = adapt(30, 80)
left_arm_end_righter = adapt(70, 0)

# right arm
right_arm_end_righter = adapt(750, 80)
right_arm_end_lefter = adapt(700, 0)

# middle_arm
middle_arm_end_lefter = adapt(350, 600)
middle_arm_end_righter = adapt(440, 600)



# FUNCTIONS
def print_info(file):
    # get video properties
    vid = cv2.VideoCapture(file)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nframes = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    duration = nframes / fps

    print(f'{file}\n'
        f'copy the following lines:\n\n'
        f'width = {width}\n'
        f'height = {height}\n\n'
        f'#left arm\n'
        f'left_corner = {left_corner}\n'
        f'middle_corner = {middle_corner}\n'
        f'left_arm_end_lefter = {left_arm_end_lefter}\n'
        f'left_arm_end_righter = {left_arm_end_righter}\n\n'
        f'#middle arm\n'
        f'right_corner = {right_corner}\n'
        f'middle_arm_end_righter = {middle_arm_end_righter}\n'
        f'middle_arm_end_lefter = {middle_arm_end_lefter}\n\n'
        f'#right arm\n'
        f'right_arm_end_righter = {right_arm_end_righter}\n'
        f'right_arm_end_lefter = {right_arm_end_lefter}\n\n'
        f'fps = {fps}\n'
        f'nframes = {nframes}\n'
        f'duration = {duration}\n')

def get_areas():
    # define all necessary areas
    left_arm = Polygon([left_corner, middle_corner, left_arm_end_righter, left_arm_end_lefter])
    right_arm = Polygon([right_corner, middle_corner, right_arm_end_lefter, right_arm_end_righter])
    middle_arm = Polygon([left_corner, right_corner, middle_arm_end_righter, middle_arm_end_lefter])
    center = Polygon([middle_corner, left_corner, right_corner])
    areas = [center, left_arm, middle_arm, right_arm]

    whole_area = Polygon([middle_corner, right_arm_end_lefter, right_arm_end_righter, 
                        right_corner, middle_arm_end_righter, middle_arm_end_lefter, 
                        left_corner, left_arm_end_lefter, left_arm_end_righter])
    whole_area = np.array(list(whole_area.exterior.coords), dtype=np.int32).reshape((-1, 1, 2))

    return areas, whole_area

def write_and_draw(file, areas):

    # write the first frame
    cap = cv2.VideoCapture(file)
    first_frame = 'first_frame.png'

    ret, frame = cap.read()
    cv2.imwrite(first_frame, frame)

    frame = image.imread(first_frame)

    for arm in areas:
            x, y = arm.exterior.xy
            plt.plot(x, y, alpha=0)
            plt.fill_between(x, y, alpha=0.5)

    plt.imshow(frame)

file = path + files[0]
print_info(file)
areas, whole_area = get_areas()
write_and_draw(file, areas)



# ADJUST BRIGHTNESS & CONTRAST
alpha = 1   # brightness: 1.0-original, <1.0-darker, >1.0-brighter
beta = 0    # contrast: 0-unchanged, <0-lower contrast, >0-higher contrast

def adjust_video(input_file, output_file):
    vid = cv2.VideoCapture(input_file)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nframes = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))

    # CROP
    x1 = left_arm_end_lefter[0]
    y1 = min(middle_arm_end_lefter[1], middle_arm_end_righter[1], left_arm_end_righter[1], right_arm_end_lefter[1])
    x2 = right_arm_end_righter[0]
    y2 = max(middle_arm_end_lefter[1], middle_arm_end_righter[1], left_arm_end_righter[1], right_arm_end_lefter[1])

    new_width, new_height = x2-x1, y2-y1                        # USE THIS WHEN     CROP     & NOT REDUCE QUALITY
    #new_width, new_height = int((x2-x1)*2/3), int((y2-y1)*2/3)  # USE THIS WHEN     CROP     &     REDUCE QUALITY
    

    # create new video
    cap = cv2.VideoCapture(input_file)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_file, fourcc, fps, (new_width, new_height)) 

    for curr_frame in tqdm(range(nframes)):
        ret, frame = cap.read()
        if ret:    

            # MASK
            if curr_frame == 0:
                white_frame = np.ones_like(frame) * 255
                mask = np.zeros_like(frame[:, :, 0])
                cv2.fillPoly(mask, [whole_area], 255)
            frame = np.where(mask[:, :, None] == 255, frame, white_frame)

            # CROP
            frame = frame[y1:y2, x1:x2]

            # BRIGHTNESS & CONTRAST
            #frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

            # RESIZE
            #frame = cv2.resize(frame, (new_width, new_height))  # must be the same dimensions as in video = cv2.VideoWriter() 

            video.write(frame)

        else:
            break

    cap.release()
    video.release()

    print(f'Done! input: {nframes} frames, output: {curr_frame+1} frames\n\n')

def main():
    for file in files:
        input_file = path + file
        output_file = input_file.replace(file_format, '') + '_edit' + file_format
        print(input_file)

        adjust_video(input_file, output_file)

#main()

