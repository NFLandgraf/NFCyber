import cv2
from matplotlib import pyplot as plt 
from matplotlib import image 
from shapely.geometry import Polygon, Point
import os



# USER INPUT
# define folder with the csv files
path = 'C:\\Users\\landgrafn\\Desktop\\kat\\'

# define values
two_upper_arms = False
left_corner = (330, 355)
middle_corner = (370, 425)
right_corner = (405, 355)




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
      f'two_upper_arms = {two_upper_arms}\n'
      f'width = {width}\n'
      f'height = {height}\n'
      f'left_corner = {left_corner}\n'
      f'middle_corner = {middle_corner}\n'
      f'right_corner = {right_corner}\n'
      f'fps = {fps}\n'
      f'nframes = {nframes}\n'
      f'duration = {duration}\n')


# if Y-maze has two upper arms or two lower arms
if two_upper_arms:
    left_arm = Polygon([left_corner, middle_corner, (middle_corner[0], 0), (0, 0), (0, left_corner[1])])
    right_arm = Polygon([right_corner, middle_corner, (middle_corner[0], 0), (width, 0), (width, right_corner[1])])
    middle_arm = Polygon([(0, left_corner[1]), left_corner, right_corner, (width, right_corner[1]), (width, height), (0, height)])
    center = Polygon([middle_corner, left_corner, right_corner])
else:
    left_arm = Polygon([left_corner, middle_corner, (middle_corner[0], height), (0, height), (0, left_corner[1])])
    right_arm = Polygon([right_corner, middle_corner, (middle_corner[0], height), (width, height), (width, right_corner[1])])
    middle_arm = Polygon([(0, left_corner[1]), left_corner, right_corner, (width, right_corner[1]), (width, 0), (0, 0)])
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