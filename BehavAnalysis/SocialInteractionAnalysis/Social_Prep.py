'''
This script is to get info about the video, the poosition of the stimuli etc.
It needs the video and the DLC file
'''

#%%
# IMPORT
import cv2
from matplotlib import pyplot as plt 
from matplotlib import image 
from shapely.geometry import Polygon, Point
from tqdm import tqdm
import numpy as np
from pathlib import Path

colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0)]

def get_files(path):
    # get all file names in directory into list
    files = [file for file in Path(path).iterdir() if file.is_file() and '.mp4' in file.name]
    
    print(f'{len(files)} videos found:')
    for file in files: 
        print(file)
    
    return files

def get_xmax_ymax(file):
    # get video properties
    vid = cv2.VideoCapture(str(file))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    return width, height

path = 'D:\\Behavior\\2025-02-26_hTauxAPP2(6m)-3m_Social'
video_file_format = '.mp4'

files = get_files(path)
width, height = get_xmax_ymax(files[0])



# USER INPUT

center_N = 135, 130
center_O = 470, 460

px_per_cm = 13.5
radius_invest = 7.3 * px_per_cm

radius_cage = 4.3 * px_per_cm
offset_invest = radius_invest/3


# FUNCTIONS
def print_info(file):
    # get video properties
    vid = cv2.VideoCapture(str(file))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nframes = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    duration = nframes / fps

    print(f'{file}\n'
        f'width = {width}\n'
        f'height = {height}\n'
        f'fps = {fps}\n'
        f'nframes = {nframes}\n'
        f'duration = {duration}\n')

def get_areas():

    # define large areas that split the whole arena
    area_large_N = Polygon([(0, 0), (width, height), (width, 0)])
    area_large_O = Polygon([(0, 0), (width, height), (0, height)])

    # define the areas of investigation
    circle_N = Point(center_N).buffer(radius_invest)
    x_N, y_N = center_N[0]+offset_invest, center_N[1]+offset_invest
    corner_N = Polygon([(x_N, y_N), (x_N, 0), (0, 0), (0, y_N)])
    area_inv_N = circle_N.union(corner_N)

    circle_O = Point(center_O).buffer(radius_invest)
    x_O, y_O = center_O[0]-offset_invest, center_O[1]-offset_invest
    corner_O = Polygon([(x_O, y_O), (x_O, height), (width, height), (width, y_O)])
    area_inv_O = circle_O.union(corner_O)

    # define the areas that cover the cage to align
    cage_N = Point(center_N).buffer(radius_cage)
    cage_O = Point(center_O).buffer(radius_cage)

    areas = [area_inv_N, area_inv_O, cage_N, cage_O]
    #areas = [area_large_N, area_large_O, area_inv_N, area_inv_O, ]

    
    return areas

def write_and_draw(file, areas):

    # write the first frame
    cap = cv2.VideoCapture(str(file))
    first_frame = 'YLine_fframe.png'

    ret, frame = cap.read()
    cv2.imwrite(first_frame, frame)

    frame = image.imread(first_frame)

    for arm in areas:
            x, y = arm.exterior.xy
            plt.plot(x, y, alpha=0)
            plt.fill_between(x, y, alpha=0.5)

    plt.imshow(frame)


def do_video(input_video_file, areas):

    # CREATE VIDEO
    # original stuff
    vid = cv2.VideoCapture(str(input_video_file))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nframes = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    
    # create new video
    output_video_file = input_video_file.with_name(input_video_file.stem + '_AnalyAreas' + video_file_format)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(str(output_video_file), fourcc, fps, (width, height)) 

    for curr_frame in tqdm(range(nframes)):
        ret, frame = vid.read()
        if ret:    


            overlay = frame.copy()
            
            for i, area in enumerate(areas):
                points_of_area = np.array(area.exterior.coords, dtype=np.int32)
                
                # draw on overlay
                cv2.fillPoly(overlay, [points_of_area], color=colors[i])

            # Blend the overlay with the frame using the alpha parameter
            frame = cv2.addWeighted(overlay, 0.5, frame, 1 - 0.5, 0)

            video.write(frame)

        else:
            break
    
    vid.release()
    video.release()



file = files[0]
print_info(file)
areas = get_areas()
write_and_draw(file, areas)
# do_video(files[0], areas)

