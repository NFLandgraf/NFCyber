# TRANSFORM .TIF INTO .JPG
import os
from PIL import Image


path = 'C:\\Users\\landgrafn\\Desktop\\aa\\'


for root, dirs, files in os.walk(path, topdown=False):
    for name in files:
        print(os.path.join(root, name))
        if os.path.splitext(os.path.join(root, name))[1].lower() == ".tiff":
            if os.path.isfile(os.path.splitext(os.path.join(root, name))[0] + ".jpg"):
                print("A jpeg file already exists")


            # If a jpeg is *NOT* present, create one from the tiff.
            else:
                outfile = os.path.splitext(os.path.join(root, name))[0] + ".jpg"
                try:
                    im = Image.open(os.path.join(root, name))
                    print("Generating jpeg")
                    #im.thumbnail(im.size)
                    print(outfile)
                    im.save(outfile, "JPEG", quality=100)
                except Exception:
                    print('nay')