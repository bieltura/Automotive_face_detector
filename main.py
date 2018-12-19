import cv2
import threading

from obj import devices
from utils import haar_face_detection as fd

# Cameras (to be written into a file)
num_cameras = 2

# Face size (square in px for CNN)
face_size = 250

# Number of cameras in the system
camera_thread = []

# Number of faces detected
faces = [None] * num_cameras


def run_camera(cam_id):

    faces[cam_id] = None

    cam = devices.Camera(cam_id)

    while True:

        frame = cam.getFrame()

        face = fd.detect_face(frame, face_size, face_size)

        if face is not None:
            faces[cam_id] = face
            cam.close()
            return


for cam in range(num_cameras):
    cam_thread = threading.Thread(target=run_camera, args=(cam,))
    camera_thread.append(cam_thread)
    cam_thread.start()

# Main program
while True:

    for cam_id, face in enumerate(faces):

        if face is not None:
            print("Face detected in camera " + str(cam_id))
            #cv2.imshow("Face detected " + str(cam_id),face)

            # Calls to the NN HERE

            # Restart camera service
            cam_thread = threading.Thread(target=run_camera, args=(cam_id,))
            camera_thread[cam_id] = cam_thread
            cam_thread.start()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
