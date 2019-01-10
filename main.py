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

end = False

# Still working on a good image
detect = cv2.imread('res/detect.png',0)


def setup(cam_id):
    cameras[cam_id] = devices.FaceCamera(cam_id)


# Runs camera captures until a face is detected
def run_face_camera(face_camera):
    face_camera.setFace(None)

    while True:

        if not end:
            frame = face_camera.captureFrame()

            if frame is not None:

                face_camera.setFace(fd.detect_face(frame, face_size, face_size))

                if face_camera.getFace() is not None:
                    face_camera.close()
                    return

        # Close the program
        else:
            face_camera.close()
            return


for cam_id in range(num_cameras):
    setup(cam_id)
    thread = threading.Thread(target=run_face_camera, args=(cameras[cam_id],))
    camera_thread.append(thread)
    thread.start()

# Main program
while True:

    for cam_id, camera in enumerate(cameras):

        frame = camera.getFrame()
        face = camera.getFace()

        if frame is not None:
            cv2.imshow(str(cam_id), cv2.resize(camera.getFrame(), (800, 450)))

        if face is not None:
            print("Face detected in camera " + str(cam_id))
            cv2.imshow(str(cam_id), cv2.resize(detect, (800, 450)))

            # Calls to the NN HERE

            # Restart camera service
            setup(cam_id)
            thread = threading.Thread(target=run_face_camera, args=(cameras[cam_id],))
            camera_thread[cam_id] = thread
            thread.start()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        end = True
        cv2.destroyAllWindows()
        break
