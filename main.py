import cv2
from obj import devices
from face_detector import CameraFaceDetector
from face_recognition import FacialRecognition
from utils import image_processing
from database import db_service as db
import glob

from threading import Thread
from nn import model
import numpy as np

# Cameras (to be written into a file)
num_cameras = 0
for camera in glob.glob("/dev/video?"):
    num_cameras += 1

# Face size (square in px for CNN)
face_size = 96

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


# Neural Net Facial recognition start
facial_recognition_thread = FacialRecognition()

# Calls to the NN HERE
print("Start thread for NN")
facial_recognition_thread.start()

#nn4_small2_pretrained = model.create_model()
#nn4_small2_pretrained.load_weights('nn/bin/nn4.small2.v1.h5')

# Main program
while True:

    for cam_id, camera in enumerate(cameras):

        frame = camera.getFrame()
        face = camera.getFace()

        if frame is not None:
            if face is not None:

                print("Face detected in camera " + str(cam_id))

                facial_recognition_thread.recognize_face(face)

                face_features = None

                # Wait for the face features to be computed
                while face_features is None:

                    camera.captureFrame()
                    frame = camera.getFrame()

                    # Set text of detecting
                    frames = image_processing.mask(frame, text)

                    cv2.imshow("Camera " + str(cam_id), cv2.resize(frames, tuple(int(x / 2) for x in camera.getDim())))

                    face_features = facial_recognition_thread.get_face_features()

                # Compare from database
                persons = db.get_all_persons()
                threshold = 0.56
                match = "unknown"
                for i, person in enumerate(persons):
                    distance = np.sum(np.square(face_features - np.fromstring(person.face_features, np.float32)))
                    if distance < threshold:
                        match = person.name
                        break

                print("Recognized as: " + str(match))
                print("")



            else:
                cv2.imshow("Camera " + str(cam_id), cv2.resize(camera.getFrame(), tuple(int(x/2) for x in camera.getDim())))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        for thread in camera_thread:
            thread.stop()
        facial_recognition_thread.stop()
        cv2.destroyAllWindows()

        break
