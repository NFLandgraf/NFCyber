#%%
import cv2
from tqdm import tqdm

# Path to the input video
input_path = "F:\\2024-10-31_m90-CA1_OF_DLC_labeled.mp4"
output_path = "F:\\2024-10-31_m90-CA1_OF_DLC_labeled_nb.mp4"

# Open the input video
cap = cv2.VideoCapture(input_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Define codec and create VideoWriter object for output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))


for frame_nb in tqdm(range(total_frames), desc='nb on frames'):
    ret, frame = cap.read()
    if not ret:
        break  # exit loop if no frames are left

    # add frame number in the top-left corner (image, text, org, font, font_scale, color, thickness, )
    cv2.putText(frame, f'{frame_nb}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 1, cv2.LINE_AA)
    out.write(frame)



cap.release()
out.release()
cv2.destroyAllWindows()