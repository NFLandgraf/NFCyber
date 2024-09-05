#%%
# IMPORT
import cv2
from matplotlib import pyplot as plt 
from matplotlib import image 
from shapely.geometry import Polygon, Point
from tqdm import tqdm
import numpy as np
from pathlib import Path


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

path = 'C:\\Users\\nicol\\NFCyber\\BehavAnalysis\\SocialInteractionAnalysis\\data'
video_file_format = '.mp4'

files = get_files(path)
width, height = get_xmax_ymax(files[0])


#%%
# USER INPUT

center_N = 133, 472
center_O = 475, 130

radius_invest = 100
radius_cage = 50
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
    area_large_N = Polygon([(0, 0), (width, height), (0, height)])
    area_large_O = Polygon([(0, 0), (width, height), (width, 0)])

    # define the areas of investigation
    circle_N = Point(center_N).buffer(radius_invest)
    x_N, y_N = center_N[0]+offset_invest, center_N[1]-offset_invest
    corner_N = Polygon([(x_N, y_N), (x_N, height), (0, height), (0, y_N)])
    area_inv_N = circle_N.union(corner_N)

    circle_O = Point(center_O).buffer(radius_invest)
    x_O, y_O = center_O[0]-offset_invest, center_O[1]+offset_invest
    corner_O = Polygon([(x_O, y_O), (x_O, 0), (width, 0), (width, y_O)])
    area_inv_O = circle_O.union(corner_O)

    # define the areas that cover the cage to align
    cage_N = Point(center_N).buffer(radius_cage)
    cage_O = Point(center_O).buffer(radius_cage)

    areas = [area_large_N, area_large_O, area_inv_N, area_inv_O, cage_N, cage_O]
    
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

file = files[0]
print_info(file)
areas = get_areas()
write_and_draw(file, areas)

