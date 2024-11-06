# TRANSFORM .TIF INTO .JPG
import os
from PIL import Image


path = 'E:\\m90\\2024-10-31-16-52-47_CA1-m90_OF_IDPS\\'


for root, dirs, files in os.walk(path, topdown=False):
    for name in files:
        print(os.path.join(root, name))
        print('\n')
        if os.path.splitext(os.path.join(root, name))[1].lower() == ".tiff":
            if os.path.isfile(os.path.splitext(os.path.join(root, name))[0] + ".jpg"):
                print("A jpeg file already exists")



            # If a jpeg is *NOT* present, create one from the tiff.
            else:
                outfile = os.path.splitext(os.path.join(root, name))[0] + ".jpg"
                
                im = Image.open(os.path.join(root, name))
                print("Generating jpeg")
                #im.thumbnail(im.size)

                print(outfile)
                im.save('C:\\Users\\landgrafn\\Desktop\\asa\\aa.png')
                print('saved')



#%%

import os
from PIL import Image
import numpy as np
import tifffile as tiff  # Optional if using tifffile for loading TIFFs

path = 'E:\\m90\\2024-10-31-16-52-47_CA1-m90_OF_IDPS\\'

for root, dirs, files in os.walk(path, topdown=False):
    for name in files:
        if os.path.splitext(os.path.join(root, name))[1].lower() == ".tiff":
            print(name)
            if os.path.isfile(os.path.splitext(os.path.join(root, name))[0] + ".jpg"):
                print("A JPEG file already exists")
            else:
                outfile = os.path.splitext(os.path.join(root, name))[0] + ".jpg"
                
                # Open the image and handle floating-point or 16-bit data
                im = Image.open(os.path.join(root, name))
                
                # Handle floating-point or 16-bit data by converting to 8-bit
                if im.mode == 'F':  # Floating-point mode
                    print("Converting from floating-point mode to 8-bit.")
                    np_image = np.array(im)
                    np_image = (255 * (np_image - np_image.min()) / np_image.ptp()).astype(np.uint8)
                    im = Image.fromarray(np_image)
                
                elif im.mode == 'I;16':  # 16-bit mode
                    print("Converting from 16-bit mode to 8-bit.")
                    np_image = np.array(im, dtype=np.uint16)
                    np_image = (np_image / 256).astype(np.uint8)
                    im = Image.fromarray(np_image)

                print("Generating JPEG")
                im.save('E:\\m90\\2024-10-31-16-52-47_CA1-m90_OF_IDPS\\MAPPPo.png')
                print('Saved as PNG:', 'C:\\Users\\landgrafn\\Desktop\\asa\\aa.png')
