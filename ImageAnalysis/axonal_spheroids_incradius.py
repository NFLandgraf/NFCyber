'''
- iterate through all pixels in an image
- at each pixel, check if neighbouring pixels (radius=0) are white
- if yes, increase the radius and check again
- save the max radius for each pixel in the image where all pixels in radius are white
'''

# Havent divided by NET % yet!!!

import os
import cv2
import numpy as np
from skimage.morphology import disk
import scipy.stats as stats
import math

path = ''


wt = ['196', '198', '200', '206', '209', '211']
app = ['706', '707', '711', '767', '768', '777']
groups = [wt, app]

NET_wt = [7.28, 5.95, 7.09, 3.52, 4.18, 3.67]
NET_app = [3.9, 2.67, 4.05, 3.25, 3.81, 9.58]


limit_radius = 200


def write_to_txt(animal, slice, radii):
    output_file = path + 'results'
    with open (output_file, 'a') as f:
        f.write(f'\nanimal: {animal}'
                f'\nslicee: {slice}'
                f'\radiiii: {radii}')

def get_files(path, common_name):
    # get files
    files = [path + file for file in os.listdir(path) 
                if os.path.isfile(os.path.join(path, file)) and
                common_name in file]
    
    return files

def load_img(file):
    # load image in grayscale and binarize (0.0/1.0) (white_px = high values, black_px = low values)
    image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    image = image // 255
    image = image.astype(np.float32)

    return image

def is_circle_white(image, center, radius):
    # fokusses a pixel and checks surrounding pixels in certain radius if they are white

    x_center, y_center = center

    # iterate through neighbours in all directions
    for angle in range(0, 360):
        x = x_center + int(radius * math.cos(math.radians(angle)))
        y = y_center + int(radius * math.sin(math.radians(angle)))

        if x < 0 or x >= image.shape[0] or y < 0 or y >= image.shape[1] or image[x, y] != 1:
            return False
        
    return True

def find_accumulations(image):
    # interates through all pxs, for each pixel append the radius where all neighbour pxs were white

    radii = []
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if image[x, y] == 1:
                radius = 1
                while is_circle_white(image, (x, y), radius):
                    radius += 1

                radii.append(radius - 1)
    
    return radii

def check_radii_dist(file, radii):

    occurences = []
    for r in range(limit_radius):
        occ = radii.count(r)
        occurences.append(occ)
    
    print(file)
    print(occurences)

    return occurences

def check_animal(name_animal):
    # checks the radii of all sclices of that animal
    files = get_files(path, name_animal)

    radii_occurences_animal = []
    for file in files:

        img = load_img(file)
        radii = find_accumulations(img)
        radii_occurences = check_radii_dist(file, radii)
        write_to_txt(name_animal, radii_occurences)

        radii_occurences_animal.append(radii_occurences)

    # averages the different radii over the slices per animal, this is the animal summary
    slices_per_animal = len(files)
    mean_radii = [np.mean([radii_occurences_animal[s][r] for s in range(slices_per_animal)]) for r in range(limit_radius)]
    write_to_txt(name_animal, 'mean', mean_radii)

    return mean_radii


wt_radii = [check_animal(animal) for animal in wt]
app_radii = [check_animal(animal) for animal in app]

