import cv2

from obj import devices
from utils import haar_face_detection as fd

# Cameras (to be written into a file)
num_cameras = 1

# Face size (square in px for CNN)
face_size = 250

# Number of cameras in the system
camera_thread = []
cameras = []

for cam in range(num_cameras):
    cameras.append(devices.Camera(cam))

while True:

    captures = []

    for cam_id, cam in enumerate(cameras):
        frame = cam.getFrame()

        frame_monitor = cv2.resize(frame, (480, 320))

        cv2.imshow('Camera ' + str(cam_id), frame_monitor)

        captures.append(frame_monitor)

        face = fd.detect_face(frame, face_size, face_size)

        if face is not -1:
            # Neural Net
            cv2.imshow('Camera ' + str(cam_id), face)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    monitor = cv2.hconcat(captures)

    # cv2.imshow('Monitor', monitor)

# When everything done, release the capture
cam.close()
cv2.destroyAllWindows()
