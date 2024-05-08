import cv2
from matplotlib import pyplot as plt 
from matplotlib import image 
from shapely.geometry import Polygon, Point
import os



# USER INPUT
# define folder with the csv files
path = 'C:\\Users\\landgrafn\\Desktop\\kat\\'

# left arm
left_corner = (330, 355)
middle_corner = (370, 425)
left_arm_end_lefter = (20, 520)
left_arm_end_righter = (60, 610)

# middle_arm
right_corner = (405, 355)
middle_arm_end_lefter = (320, 0)
middle_arm_end_righter = (410, 0)

# right arm
right_arm_end_righter = (720, 530)
right_arm_end_lefter = (680, 610)




# get all file names in directory into list
files = [file for file in os.listdir(path) 
             if os.path.isfile(os.path.join(path, file)) and
             'mp4' in file]

file = path + files[0]


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



left_arm = Polygon([left_corner, middle_corner, left_arm_end_righter, left_arm_end_lefter])
right_arm = Polygon([right_corner, middle_corner, right_arm_end_lefter, right_arm_end_righter])
middle_arm = Polygon([left_corner, right_corner, middle_arm_end_righter, middle_arm_end_lefter])
center = Polygon([middle_corner, left_corner, right_corner])

areas = [left_arm, middle_arm, right_arm]



def write_and_draw(file, areas):

    # write the first frame
    cap = cv2.VideoCapture(file)
    first_frame = 'first_frame.png'
    ret, frame = cap.read()
    cv2.imwrite(first_frame, frame)

    # draw
    data = image.imread(first_frame)

    for arm in areas:
            x, y = arm.exterior.xy
            plt.plot(x, y, alpha=1)
            plt.fill_between(x, y, alpha=0.5)

    

    plt.imshow(data)

write_and_draw(file, areas)
