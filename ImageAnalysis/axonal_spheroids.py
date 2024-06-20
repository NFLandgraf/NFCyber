#%%
'''
- lay a circle with a fixed radius on each pixel 
- save the pixels where the number of white pixel in that circle is above a threshold
- scaling 20x: 0.32um/px, 10x: 0.65um/px, 2.5x: 2.58um/px

'''

import os
import cv2
import numpy as np
from skimage.morphology import disk
import scipy.stats as stats
import math



path = 'C:\\Users\\landgrafn\\NFCyber\\ImageAnalysis\\data\\'


min_AS_diam_um = 4
threshold = 0.85  # in decimals

wt = ['211', '196', '206', '209']
wt = ['209']
app = ['706', '707', '711', '768', '777']







scaling_measure = 0.32  # um/px
radius = int((min_AS_diam_um / scaling_measure) / 2)
print(f'radius in um/px: {min_AS_diam_um} / {radius}\n'
      f'threshold: {threshold}\n')

px_in_kernel = [0,5,13,29,49,81,113,149,197,253,317]
threshold = int(threshold * px_in_kernel[radius])

groups = [wt]

NET_wt = [7.28, 5.95, 7.09, 3.52, 4.18, 3.67]
NET_app = [3.9, 2.67, 4.05, 3.25, 3.81, 9.58]

def get_files(path, common_name):
    # get files
    files = [path + file for file in os.listdir(path) 
                if os.path.isfile(os.path.join(path, file)) and
                common_name in file]
    
    return files

def show_image(img):
    cv2.imshow('Mask', img.astype(np.uint8) * 255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def get_accumulations(pic, radius, threshold):

    # load image in grayscale and binarize (0.0/1.0) (white_px = high values, black_px = low values)
    image = cv2.imread(pic, cv2.IMREAD_GRAYSCALE)
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    image = image // 255
    image = image.astype(np.float32)

    # Create a disk-shaped kernel [0,1,0,...], lay kernel above each px and sum the '1' values around px 
    kernel = disk(radius)
    convolution_result = cv2.filter2D(image, cv2.CV_32F, kernel)    # sum of surrounding px values
    mask = convolution_result > threshold   # saying True/False if sum(neighbour_px) > threshold
    show_image(mask)

    # get results
    nb_white_acc, labels_im = cv2.connectedComponents(mask.astype(np.uint8))
    nb_white_acc = nb_white_acc -1
    nb_white_px = mask.sum()

    return nb_white_acc, nb_white_px

def check_animal(common_name, radius, threshold):
    # go through all files of an animal and return the animals' mean
    print(common_name)
    files = get_files(path, common_name)
    acc_list, px_list = zip(*(get_accumulations(file, radius, threshold) for file in files))

    return np.mean(acc_list), np.mean(px_list)

def check_group(animal_list, radius, threshold):
    # go through animal group (e.g. WT) and return its mean values
    
    acc_means, px_means = zip(*(check_animal(animal, radius, threshold) for animal in animal_list))

    return acc_means, px_means

'''
def calc_stats(radius, threshold):

    # devide by the %area of NET to normalize it
    results_wt = check_group(wt, radius, threshold) / NET_wt
    results_app = check_group(app, radius, threshold) / NET_app
    
    stat, pvalue = stats.ttest_ind(a=results_wt, b=results_app, equal_var=True)

    # only return if pvalue is not nan
    if math.isnan(pvalue):
        return None
    else:
        return round(pvalue, 5)

def main():

    all_values = []
    i = 0
    for radius in radii:
        for threshold in thresholds:
            pvalue = calc_stats(radius, threshold)

            if pvalue != None:
                all_values.append([pvalue, radius, threshold])

            print(i)
            i+=1

    return all_values

'''

# divide by the %area of NET to normalize it!
for group in groups:
    acc_means, px_means = check_group(group, radius, threshold)
    print(group)
    print(acc_means)
    print(px_means)
    
