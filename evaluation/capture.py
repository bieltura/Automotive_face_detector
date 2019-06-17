import cv2
from devices import cameras
import numpy as np

stereo = True

frame_width = 800
frame_height = int(frame_width * 9 / 16)

if stereo:
    num_cameras = 2
else:
    num_cameras = 1

cam = [None] * num_cameras

# Setup the cameras
for cam_id in range(num_cameras):
    cam[cam_id] = cameras.Camera(cam_id)
    cam[cam_id].start()

img_counter = 26

while True:
    frame_right = cam[0].getFrame()

    if frame_right is not None:

        frame = None

        if stereo:
            frame_left = cam[1].getFrame()

            if frame_left is not None:
                frame = np.concatenate((frame_right, frame_left), axis=1)
        else:
            frame = frame_right

        if frame is not None:
            cv2.imshow("Camera capture", cv2.resize(frame, (frame_width*2, frame_height)))

    k = cv2.waitKey(1)

    # ESC pressed
    if k % 256 == 27:
        print("Escape hit, closing...")
        break

    # SPACE pressed
    elif k % 256 == 32:
        img_name = "img/{0}_r.png".format(img_counter)
        if stereo:
            cv2.imwrite("img/{0}_r.png".format(img_counter), frame_right)

            cv2.imwrite("img/{0}_l.png".format(img_counter), frame_left)
        else:

            cv2.imwrite("img/{0}.png".format(img_counter), frame)

        print("{0} image captured.".format(img_name))

        img_counter += 1

# Camera thread stop
for camera in cam:
    camera.close()

cv2.destroyAllWindows()