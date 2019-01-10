import cv2
import threading

from obj import devices
from utils import haar_face_detection as fd

# Cameras (to be written into a file)
num_cameras = 2

# Face size (square in px for CNN)
face_size = 250

# Number of cameras in the system
cameras = [None] * num_cameras
camera_thread = []

# Number of faces detected
faces = [None] * num_cameras
frames = [None] * num_cameras

end = False


def run_camera(cam_id):
    cameras[cam_id] = devices.Camera(cam_id)

    while True:

        if not end:
            frames[cam_id] = cameras[cam_id].getFrame()

            faces[cam_id] = fd.detect_face(frames[cam_id], face_size, face_size)

            if faces[cam_id] is not None:
                cameras[cam_id].close()
                return

        # Close the program
        else:
            cameras[cam_id].close()
            return


for cam_id in range(num_cameras):
    thread = threading.Thread(target=run_camera, args=(cam_id,))
    camera_thread.append(thread)
    thread.start()

# Main program
while True:

    for cam_id in range(num_cameras):

        if frames[cam_id] is not None:
            cv2.imshow(str(cam_id), cv2.resize(frames[cam_id], (800, 450)))

        if faces[cam_id] is not None:
            print("Face detected in camera " + str(cam_id))

            # Calls to the NN HERE

            # Remove the face
            faces[cam_id] = None

            # Restart camera service
            thread = threading.Thread(target=run_camera, args=(cam_id,))
            camera_thread[cam_id] = thread
            thread.start()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        end = True
        cv2.destroyAllWindows()
        break
