'''''
From Y_Line
1. saves every frame from a video
2. draws on every saved frame
3. makes a vvideo from all frames
'''''


# CREATE VIDEO
def write_all_frames_from_video(input_file, output_folder):
    # takes video and creates png for every frame in video
    os.makedirs(output_folder, exist_ok=True)

    vidcap = cv2.VideoCapture(input_file)
    nframes = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    success, image = vidcap.read()

    print('\nwrite_all_frames_from_video')
    count = 0
    pbar = tqdm(total=int(nframes))
    while success:
        cv2.imwrite(output_folder + '\\frame%d.png' % count, image)
        success,image = vidcap.read()
        count += 1
        pbar.update(1)

def draw_and_write(input_folder, output_folder, areas, positions):
    # takes pngs and draws polygon onto it, if mouse in that position
    os.makedirs(output_folder, exist_ok=True)

    images = [file for file in os.listdir(input_folder) if file.endswith('.png')]

    # sort the list of strings based on the int in the string
    sorted_intimages = sorted([int(re.sub("[^0-9]", "", element)) for element in images])
    sorted_images = [f'frame{element}.png' for element in sorted_intimages]

    print('\ndraw_and_write')
    for i in tqdm(range(len(sorted_images))):
        
        image_path = os.path.join(input_folder, sorted_images[i])
        data = cv2.imread(image_path)

        for arm, area in enumerate(areas):
            if arm == positions[i]:
                points = np.array(area.exterior.coords, dtype=np.int32)
                cv2.polylines(data, [points], isClosed=True, color=(0, 0, 255), thickness=2)
        
        output_path = os.path.join(output_folder, f"frame{i}.png")
        cv2.imwrite(output_path, data)

def video_from_frames(input_folder, path):
    # created video from all frames with drawn polygons
    output = os.path.join(path, f'{files[0]}_edit.mp4')

    images = [file for file in os.listdir(input_folder) if file.endswith('.png')]

    # sort the list of strings based on the int in the string
    sorted_intimages = sorted([int(re.sub("[^0-9]", "", element)) for element in images])
    sorted_images = [f'frame{element}.png' for element in sorted_intimages]

    # setting the video properties according to first image 
    frame0 = cv2.imread(os.path.join(input_folder, sorted_images[0]))
    height, width, layers = frame0.shape 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output, fourcc, fps, (width, height)) 

    # Appending the images to the video
    print('\nvideo_from_frames')
    for i in tqdm(range(len(sorted_images))):
        img_path = os.path.join(input_folder, sorted_images[i])
        img = cv2.imread(img_path)
        video.write(img)

    cv2.destroyAllWindows()
    video.release()

def create_drawn_video(video_file, folder_framesfromvideo, folder_draw_on_frames, path, areas, positions):
    write_all_frames_from_video(video_file, folder_framesfromvideo)
    draw_and_write(folder_framesfromvideo, folder_draw_on_frames, areas, positions)
    video_from_frames(folder_draw_on_frames, path)

def do_video(all_positions, file_name):
    video_file = path + file_name
    vid = cv2.VideoCapture(video_file)
    nframes = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    folder_framesfromvideo = path + '\\frames_from_video'
    folder_draw_on_frames = path + '\\edited_frames'
    create_drawn_video(video_file, folder_framesfromvideo, folder_draw_on_frames, path, areas, all_positions)

    print('video done')
    

# USER INPUT 2
videofile_name = '\\513_TSPO-KO_NaiveDLC_resnet50_TopoViewMouseMar22shuffle1_600000_filtered_labeled.mp4'
do_video(all_positions, videofile_name)
