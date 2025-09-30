#%%
import pandas as pd
import cv2

def get_video_dims(file_video):
    cap = cv2.VideoCapture(file_video)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return (width, height)



def do(file_video, file_main, data_prefix):

    width, height = get_video_dims(file_video)

    df = pd.read_csv(file_main, index_col='Time', low_memory=False)

    # transform the coordinates into mm to normalize for camera reposition between recordings (we have the same dimension in mm during video_processing so all good)
    if 'OF' in data_prefix:
        scale_x = 460 / width   # mm/px
        scale_y = 460 / height  # mm/px
        print('OF! for transformation px into mm ')
    elif 'YMaze' in data_prefix:
        scale_x = 580 / width   # mm/px
        scale_y = 485 / height  # mm/px
        print('YMaze! for transformation px into mm ')
    else:
        raise ValueError('Bro, Transformation px into mm doesnt know which arena you use')

    for col in df.columns:
        if col.endswith('_x'):
            df[col] = df[col] * scale_x
        elif col.endswith('_y'):
            df[col] = df[col] * scale_y
    
    return df


data_prefix = "D:\\CA1Dopa_Miniscope\\Post_f48\\CA1Dopa_Longitud_f48_Post3_YMaze_"
file_main    =   data_prefix + 'main.csv'
file_video  =   data_prefix + 'video_DLC.mp4'




main_df = do(file_video, file_main, data_prefix)
main_df.to_csv(data_prefix + 'main_mm.csv')







