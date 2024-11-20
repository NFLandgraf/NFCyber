#%%
import os 
import cv2 
import re
from tqdm import tqdm

# frames folder
path = "C:\\Users\\landgrafn\Desktop\\m90_data\\2024-10-31-16-52-47_CA1-m90_OF_DLC_pics\\"
common_name = '.jpg'
fps = 30
output = os.path.join(path, 'video.mp4')

# get all file names in directory into list
images = [file for file in os.listdir(path) 
          if os.path.isfile(os.path.join(path, file)) and 
          common_name in file]

# sort images according to numbers in name
def num_sort(file_name):
    return int(re.findall(r'\d+', file_name)[0])
images.sort(key=num_sort) 
print(f'{len(images)} files found in path directory')

# set the video properties according to first image 
frame0 = cv2.imread(os.path.join(path, images[0])) 
height, width, layers = frame0.shape 
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(output, fourcc, fps, (width, height)) 

# append the images to the video one by one 
for i in tqdm(range(len(images))): 
    frame = cv2.imread(os.path.join(path, images[i]))
    
	# add frame number in the top-left corner (image, text, org, font, font_scale, color, thickness, )
    # cv2.putText(frame, i, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, cv2.LINE_AA)

    video.write(frame) 

cv2.destroyAllWindows()
video.release()
