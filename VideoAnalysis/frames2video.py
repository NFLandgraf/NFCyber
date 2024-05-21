#%%
# IMPORT

import os 
import cv2 
from PIL import Image 
import re



#%%
# CREATE VIDEO FROM FRAMES


# Folder which contains all the images from which video is to be generated 
path = "C:\\Users\\nicol\\NFCyber\\VideoAnalysis\\data\\"
common_name = '.png'
fps = 30
output = path + 'video.mp4'

# get all file names in directory into list
images = [file for file in os.listdir(path) 
             if os.path.isfile(os.path.join(path, file)) and 
			 common_name in file]

# sort images according to numbers in name
def num_sort(file_list):
	return list(map(int, re.findall(r'\d+', file_list)))[0]
images.sort(key=num_sort) 
print(f'{len(images)} files found in path directory')

# setting the video properties according to first image 
frame0 = cv2.imread(images[0]) 
height, width, layers = frame0.shape 
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(output, fourcc, fps, (width, height)) 

# Appending the images to the video one by one 
for img in images: 
	video.write(img) 

cv2.destroyAllWindows()
video.release()