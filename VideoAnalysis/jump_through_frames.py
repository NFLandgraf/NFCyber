#%%
import cv2

video_path = r"Y:\_proj_Nico\AGG_Behav\Solomon\Training\2025-02-12_hTauxAPP1(3m)_RI3_m251_Test_edit_fps_solomon.avi"


cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
current_frame = 0

print(f"Controls:")
print(f" ->  D or Right Arrow: Next Frame")
print(f" ->  A or Left Arrow: Previous Frame")
print(f" ->  S: Jump +100 frames")
print(f" ->  W: Jump -100 frames")
print(f" ->  ESC: Close window")

while True:
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    ret, frame = cap.read()
    
    if not ret:
        break
    cv2.imshow('Video Frame-by-Frame', frame)
    print(f"Frame: {current_frame}/{total_frames}", end="\r")

    # Wait for keyboard input
    key = cv2.waitKey(0) & 0xFF
    
    if key == 27: # ESC key to exit
        break
    elif key == ord('d'): # Next frame
        current_frame = min(current_frame + 1, total_frames - 1)
    elif key == ord('a'): # Previous frame
        current_frame = max(current_frame - 1, 0)
    elif key == ord('s'): # Jump forward
        current_frame = min(current_frame + 100, total_frames - 1)
    elif key == ord('w'): # Jump backward
        current_frame = max(current_frame - 100, 0)

cap.release()
cv2.destroyAllWindows()