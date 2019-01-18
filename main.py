import cv2
from obj import devices
from face_detector import CameraFaceDetector
from face_recognition import FacialRecognition
from utils import image_processing
import glob

# Cameras (to be written into a file)
num_cameras = 0
for camera in glob.glob("/dev/video?"):
    num_cameras += 1

# Face size (square in px for CNN)
face_size = 250

# Number of cameras in the system
cameras = [None] * num_cameras
camera_thread = []

# Still working on a good image
text = cv2.imread('res/text_detect.png')

# Setup the cameras
for cam_id in range(num_cameras):
    cameras[cam_id] = devices.FaceCamera(cam_id)
    thread = CameraFaceDetector(cameras[cam_id], face_size)
    camera_thread.append(thread)
    thread.start()


# Main program
while True:

    for cam_id, camera in enumerate(cameras):

        frame = camera.getFrame()
        face = camera.getFace()

        if frame is not None:
            if face is not None:

                # Neural Net Facial recognition start
                facial_recognition_thread = FacialRecognition(face)

                print("Face detected in camera " + str(cam_id))

                # Set text of detecting
                frames = image_processing.mask(frame, text)

                cv2.imshow("Camera " + str(cam_id), cv2.resize(frames, tuple(int(x/2) for x in camera.getDim())))
                #cv2.imshow("Face " + str(cam_id), face)

                # Calls to the NN HERE
                #facial_recognition_thread.start()

            else:
                cv2.imshow("Camera " + str(cam_id), cv2.resize(camera.getFrame(), tuple(int(x/2) for x in camera.getDim())))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        for thread in camera_thread:
            thread.stop()
        cv2.destroyAllWindows()

        break
