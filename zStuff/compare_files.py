#%%

import os
import cv2
from pathlib import Path

folder_new = Path(r"D:\new\Behav_3_Mask")
folder_old = Path(r"W:\_proj_CA1Dopa\CA1Dopa_Longitud\Analysis_new\Behav_4_mask")
ext = ".mp4"


def video_info(path):
    size = path.stat().st_size / 1024
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()
    return frames, size


for vid_new in folder_new.glob(f"*{ext}"):
    print(vid_new)
    vid_old = folder_old / vid_new.name

    if not vid_old.exists():
        print(f"🔴 {vid_new.name}: NOT FOUND in second folder")
        continue

    frames_new, size_new = video_info(vid_new)
    frames_old, size_old = video_info(vid_old)

    if frames_new == frames_old and size_new == size_old:
        print('✅ same same')
    else:
        print(f'🔴 {frames_new} vs {frames_old}')
        print(f'🔴 {size_new} vs {size_old}')


