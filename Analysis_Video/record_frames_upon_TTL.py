import cv2
import time
from pypylon import pylon



timeout = 10000 #ms
fps_grabbing = 100  # maximum for this cam is 164fps

output_folder = 'Pylon_Project\\Output\\'



# Create an instance of the camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()

# Configure the camera for external trigger
camera.TriggerSelector.SetValue('FrameStart')
camera.TriggerMode.SetValue("On")  # Enable external triggering
camera.TriggerSource.SetValue("Line1")  # Set to use the camera's digital input line (Line1 for TTL trigger)

# Set the resolution (Width and Height)
#camera.Width.SetValue(width)
#camera.Height.SetValue(height)
#camera.ExposureTime.SetValue(exposure_time)

# Set the frame rate of how many fps are grabbed (not captured\saved!!)
#camera.AcquisitionFrameRateEnable.SetValue(True)
#camera.AcquisitionFrameRate.SetValue(fps_grabbing)



def record_on_ttl():

    fps = camera.ResultingFrameRate.GetValue()
    print(f"Current FPS: {fps}")

    # Start grabbing images (camera waits for trigger)
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    # Define a counter to save frames
    frame_count = 0

    print("Waiting for TTL trigger...")

    #cv2.namedWindow('Last Frame', cv2.WINDOW_NORMAL)


    while True:
        grab_result = camera.RetrieveResult(timeout, pylon.TimeoutHandling_ThrowException)  # Timeout is 5 seconds

        if grab_result.GrabSucceeded():

            # Check if the trigger was received
            if camera.TriggerSource.GetValue() == "Line1" and camera.TriggerMode.GetValue() == "On":
                frame = grab_result.GetArray()

                filename = f'{output_folder+str(frame_count)}.png'
                cv2.imwrite(filename, frame)

                print(f"Frame saved {frame_count}")
                
                frame_count += 1

            else:
                print("No valid trigger received, waiting for TTL pulse...")
        else:
            print("Failed to grab image.")
        
        #cv2.imshow('Last Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Check for a keypress to stop recording (press 'q' to quit)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Release the grab result
        grab_result.Release()

    # Stop grabbing and release resources
    camera.StopGrabbing()
    camera.Close()

    print(f"Stopped recording. Total frames captured: {frame_count}")

    cv2.destroyAllWindows()  # Close the OpenCV window




record_on_ttl()
