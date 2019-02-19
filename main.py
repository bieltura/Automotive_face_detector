import cv2
from obj import devices
from face_detector import CameraFaceDetector
from face_recognition import FacialRecognition
from utils import image_processing
from database import db_service as db
import glob

from nn import model
import numpy as np

# Check number of cameras in Linux distribution cameras
num_cameras = 0
for camera in glob.glob("/dev/video?"):
    num_cameras += 1

# Face size (square in px for CNN)
face_size = 224

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
#facial_recognition_thread = FacialRecognition()

# Calls to the NN HERE
#print("Start thread for NN")
#facial_recognition_thread.start()

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
                #cv2.imshow("Face " + str(cam_id), face)

                landmarks = camera.getLandmarks()

                if landmarks is not None:
                    # Hard code for the landmarks
                    for i, (x,y) in enumerate(landmarks):

                        # Small 5 shape landmakrs
                        if i in [36, 39, 45, 42, 33]:

                            # Affine transformation
                            if i in [36, 45, 33]:
                                cv2.circle(frame, (x, y), 8, (255, 255, 255), -1)
                                cv2.putText(frame, 'Affine transform', (camera.getDim()[0]-450,camera.getDim()[1]-200),
                                                                        cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255))

                            cv2.circle(frame, (x, y), 5, (255, 0, 255), -1)
                            cv2.putText(frame, '5 point shape predictor', (camera.getDim()[0]-450, camera.getDim()[1]-150),
                                        cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 255))
                        else:
                            cv2.circle(frame, (x, y), 4, (192, 162, 103), -1)
                            cv2.putText(frame, '68 point shape predictor', (camera.getDim()[0]-450, camera.getDim()[1]-100),
                                        cv2.FONT_HERSHEY_TRIPLEX, 1, (192, 162, 103))

                cv2.imshow("Camera " + str(cam_id), cv2.resize(frame, tuple(int(x / 2) for x in camera.getDim())))

                """
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
                """

            else:
                cv2.imshow("Camera " + str(cam_id), cv2.resize(frame, tuple(int(x/2) for x in camera.getDim())))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        for thread in camera_thread:
            thread.stop()
        #facial_recognition_thread.stop()
        cv2.destroyAllWindows()

        break
