import cv2
from pypylon import pylon
import time

width = 500
height = 500
exposure_time = 10000
period = 1

output = 'Pylon_Project\\Output\\'


def save_frames():
    # Create an instance of the camera
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()

    camera.Width.SetValue(width)
    camera.Height.SetValue(height)
    camera.ExposureTime.SetValue(exposure_time)

    print("Recording video...")

    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    frame_count = 0

    while True:
        grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)  # Timeout is 5 seconds

        if grab_result.GrabSucceeded():
            frame = grab_result.GetArray()

            print(f"Frame {frame_count}")
            cv2.imwrite(f'{output+str(frame_count)}.png', frame)
            cv2.imshow('Video Recording', frame)
            frame_count += 1

        else:
            print("Failed to grab image")


        time.sleep(period)


        # Check for keypress to stop recording (press 'q' to quit)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Release the grab result
        grab_result.Release()

    camera.StopGrabbing()
    camera.Close()

    cv2.destroyAllWindows()

    print(f"Total frames captured: {frame_count}")


save_frames()
