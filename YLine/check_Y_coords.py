import cv2
from matplotlib import pyplot as plt 
from matplotlib import image 
from shapely.geometry import Polygon, Point

file = 'data\\Y.mp4'

top_corner = (350, 180)
left_corner = (315, 250)
right_corner = (390, 250)
corners = [top_corner, left_corner, right_corner]


# get video properties
vid = cv2.VideoCapture(file)

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


left_arm = Polygon([left_corner, top_corner, (top_corner[0], 0), (0, 0), (0, left_corner[1])])
right_arm = Polygon([right_corner, top_corner, (top_corner[0], 0), (width, 0), (width, right_corner[1])])
bottom_arm = Polygon([(0, left_corner[1]), left_corner, right_corner, (width, right_corner[1]), (width, height), (0, height)])
center_arm = Polygon([top_corner, left_corner, right_corner])

areas = [left_arm, right_arm, bottom_arm]



def write_and_draw(file, areas, corners):

    # write the first frame
    cap = cv2.VideoCapture(file)
    first_frame = 'first_frame.png'
    ret, frame = cap.read()
    cv2.imwrite(first_frame, frame)

    # draw
    data = image.imread(first_frame)

    for arm in areas:
            x, y = arm.exterior.xy
            plt.plot(x, y)
            plt.fill_between(x, y, alpha=0.5)

    for corner in corners:
        plt.plot(corner[0], corner[1], marker='.', color='black')

    plt.imshow(data)

write_and_draw(file, areas, corners)