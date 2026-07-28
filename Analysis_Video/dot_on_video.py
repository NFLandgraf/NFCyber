#%%
import cv2

input_path = r"D:\CA1Dopa_2pTube_2025-07-30_1001_Airpuff_alt_Ch0_prepro_MotCorr120.avi"
output_path = r"D:\CA1Dopa_2pTube_2025-07-30_1001_Airpuff_alt_Ch0_prepro_MotCorr120_20fps.mp4"
new_fps = 20

cap = cv2.VideoCapture(input_path)
orig_fps = cap.get(cv2.CAP_PROP_FPS)
frame_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
skip     = int(orig_fps / new_fps)

out = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*'mp4v'), new_fps, (frame_w, frame_h))

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % skip == 0:
        out.write(frame)
    frame_idx += 1

cap.release()
out.release()
print("✅ Saved reduced-fps video:", output_path)


#%%

import cv2

input_path = r"D:\CA1Dopa_2pTube_2025-07-30_1001_Airpuff_alt_Ch0_prepro_MotCorr120_20fps.mp4"
output_path = r"D:\CA1Dopa_2pTube_2025-07-30_1001_Airpuff_alt_Ch0_prepro_MotCorr120_20fps_dot.mp4"

cap = cv2.VideoCapture(input_path)
fps = cap.get(cv2.CAP_PROP_FPS)
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

# red dot parameters
dot_pos = (10, 10)
dot_radius = 10
dot_color  = (0, 0, 255)
dot_starts = [53, 108, 158, 210, 270]
dot_duration = 2

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # draw red dot
    for s in dot_starts:
        if s <= frame_idx < s + dot_duration:
            cv2.circle(frame, dot_pos, dot_radius, dot_color, -1)


    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
print("✅ Saved video with frame numbers:", output_path)

