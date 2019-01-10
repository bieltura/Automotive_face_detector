import cv2
import threading
import numpy as np

from obj import devices
from utils import haar_face_detection as fd

# Cameras (to be written into a file)
num_cameras = 2

# Face size (square in px for CNN)
face_size = 250

# Number of cameras in the system
cameras = [None] * num_cameras
detection_mask = [None] * num_cameras
camera_thread = []

end = False

# Still working on a good image
text = cv2.imread('res/text_detect.png')

def setup(cam_id):
    cameras[cam_id] = devices.FaceCamera(cam_id)
    print(cameras[cam_id].getDim())


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

            detection_text = cv2.resize(text, camera.getDim())

            detection_text_gray = cv2.cvtColor(detection_text, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(detection_text_gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            # Now black-out the area of logo in ROI
            img1_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)

            # Take only region of logo from logo image.
            img2_fg = cv2.bitwise_and(detection_text, detection_text, mask=mask)

            # Put text in ROI and modify the main image
            frame = cv2.add(img1_bg, img2_fg)

            cv2.imshow(str(cam_id), cv2.resize(frame, (800, 450)))

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