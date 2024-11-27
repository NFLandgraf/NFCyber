#%%
import cv2
from tqdm import tqdm

addo = 150
por = 15
peep1 = 460
peep2 = peep1 + addo
peep3 = peep2 + addo
peep4 = peep3 + addo
peep5 = peep4 + addo


frames_to_modify =  set(range(peep1,peep1+por))|set(range(peep2,peep2+por))|set(range(peep3,peep3+por))|set(range(peep4,peep4+por))|set(range(peep5,peep5+por))



# Path to the input video
input_path = "C:\\Users\\landgrafn\\Desktop\\LC\\2024-09-18-18-35-37_LC_m379_FC_motcorr_rate64.mp4"
output_path = "C:\\Users\\landgrafn\\Desktop\\LC\\2024-09-18-18-35-37_LC_m379_FC_motcorr_rate64_circle.mp4"

# Open the input video
cap = cv2.VideoCapture(input_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(total_frames)

# Define codec and create VideoWriter object for output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))


for frame_nb in tqdm(range(total_frames), desc='nb on frames'):
    ret, frame = cap.read()
    if not ret:
        break  # exit loop if no frames are left

    # add frame number in the top-left corner (image, text, org, font, font_scale, color, thickness, )
    #cv2.putText(frame, f'{frame_nb}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 1, cv2.LINE_AA)
    
    # add sth on specific frames
    if frame_nb in frames_to_modify:
        cv2.circle(frame, center=(20,20) , radius=20, color = (0, 0, 255) , thickness=-1)

    
    out.write(frame)



cap.release()
out.release()
cv2.destroyAllWindows()