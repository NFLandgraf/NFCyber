import cv2
from pypylon import pylon
import threading
import os
import time



timeout = 10000 # after how many ms not receiving any TTL is the cam stopping
new_folder_name = 'Testy'



# Create an instance of the camera
cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
cam.Open()
print("Using device ", cam.GetDeviceInfo().GetModelName())


# Configure the camera for external trigger
cam.TriggerSelector.SetValue('FrameStart')
cam.TriggerMode.SetValue("On")  # enable external triggering
cam.TriggerSource.SetValue("Line1")  # set to use the camera's digital input line (Line1 for TTL trigger)



# Set the resolution (Width and Height)
#camera.Width.SetValue(width)
#camera.Height.SetValue(height)
#camera.ExposureTime.SetValue(exposure_time)


def create_dir(new_folder_name):
    # creates new folder for all frames

    try:
        parent_folder = 'C:\\Users\\Admin\\Nico_SSD_Cam\\'
        output_folder = parent_folder + new_folder_name

        os.mkdir(output_folder)
        print(f"Directory '{output_folder}' created successfully")

        return output_folder
    
    except Exception as e:
        print(f"An error occurred: {e}")

def save_frame(frame, frame_count, save_count, output_folder):
    filename = f'{output_folder}//_{frame_count}.jpg'
    cv2.imwrite(filename, frame)
    #print(f"{frame_count} frame saved.")
    save_count += 1

def record_on_ttl(output_folder):

    # Start grabbing images (camera waits for trigger)
    cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    frame_count = 0
    save_count = 0
    print("Waiting for TTL trigger...")

    # to see the frames live
    cv2.namedWindow('Last Frame', cv2.WINDOW_NORMAL)


    while True:
        # the cam only captures a frame upon TTL
        grab_result = cam.RetrieveResult(timeout, pylon.TimeoutHandling_ThrowException)  # after how many ms of nor receiving a TTL is the cam stopping

        if grab_result.GrabSucceeded():
            frame_count += 1

            frame = grab_result.GetArray()

            # Offload the frame saving to a separate thread
            threading.Thread(target=save_frame, args=(frame, frame_count, save_count, output_folder)).start()            
            frame_count += 1

            
        else:
            print("Failed to grab image.")

        # to see the frames live
        cv2.imshow('Last Frame', frame)    # if you want to see the frames

        
        # Check for a keypress to stop recording (press 'q' to quit)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Release the grab result
        grab_result.Release()

    # Stop grabbing and release resources
    cam.StopGrabbing()
    cam.Close()

    print(f"Stopped recording. Total frames captured: {frame_count}")

    cv2.destroyAllWindows()

    print(len(frame_count))
    print(len(save_count))



output_folder = create_dir(new_folder_name)
record_on_ttl(output_folder)
