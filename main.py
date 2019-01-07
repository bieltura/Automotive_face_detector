import cv2
import threading

from obj import devices
from utils import haar_face_detection as fd

# Cameras (to be written into a file)
num_cameras = 1

# Face size (square in px for CNN)
face_size = 250

# Number of cameras in the system
camera_thread = []

# Number of faces detected
faces = [None] * num_cameras

frames = [None] * num_cameras

end = False


def run_camera(cam_id):

    faces[cam_id] = None
    frames[cam_id] = None

    camera_device = devices.Camera(cam_id)

    while True:

        if not end:

            frame = camera_device.getFrame()

            frames[cam_id] = frame

            face = fd.detect_face(frame, face_size, face_size)

            if face is not None:
                faces[cam_id] = face
                camera_device.close()
                return

        else:
            camera_device.close()
            return


for cam_id in range(num_cameras):
    thread = threading.Thread(target=run_camera, args=(cam_id,))
    camera_thread.append(thread)
    thread.start()

# Main program
while True:

    for cam_id, face in enumerate(faces):

        if frames[cam_id] is not None:
            cv2.imshow(str(cam_id),frames[cam_id])

        if face is not None:
            print("Face detected in camera " + str(cam_id))
            #cv2.imshow("Face detected " + str(cam_id),face)

            # Calls to the NN HERE

            # Restart camera service
            thread = threading.Thread(target=run_camera, args=(cam_id,))
            camera_thread[cam_id] = thread
            thread.start()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        end = True
        cv2.destroyAllWindows()
        break
