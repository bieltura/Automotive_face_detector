import cv2
import threading
import sys

from obj import devices
from utils import haar_face_detection as fd

# Cameras (to be written into a file)
num_cameras = 2

# Face size (square in px for CNN)
face_size = 250

# Number of cameras in the system
camera_thread = []

def run_camera(cam_id):

    cam = devices.Camera(cam_id)

    while True:

        captures = []

        frame = cam.getFrame()

        frame_monitor = cv2.resize(frame, (480, 320))

        #cv2.imshow('Camera ' + str(cam_id), frame_monitor)

        captures.append(frame_monitor)

        face = fd.detect_face(frame, face_size, face_size)

        if face is not -1:
            # Neural Net
            print("face detected in camera" + str(cam_id))
            #cv2.imshow('Camera ' + str(cam_id), face)

    # When everything done, release the capture
    cam.close()
    #cv2.destroyAllWindows()


for cam in range(num_cameras):
    cam_thread = threading.Thread(target=run_camera, args=(cam,))
    camera_thread.append(cam_thread)
    cam_thread.start()

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):

        # Kill threads?
        #for thread in cam_thread:

        sys.exit(0)
