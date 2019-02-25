import cv2
import glob
import time
import numpy as np
from obj import devices
from face_detector import CameraFaceDetector
from face_recognition import FacialRecognition
from utils import demo
from database import db_service as db
from utils.port_serial import arduino_serial

# Demonstration values:
face_detector_demo = False
face_recognition_demo = False
fps_demo = False
serial_demo = True

# Check number of cameras in Linux distribution cameras
num_cameras = 0
for camera in glob.glob("/dev/video?"):
    num_cameras += 1

# Check number of port serials in Linux distribution
if serial_demo:
    for port, camera in enumerate(glob.glob("/dev/ttyACM?")):
        arduino = arduino_serial('/dev/ttyACM'+str(port))

# Face size (square in px for CNN)
face_size = 96

# Number of cameras in the system
cameras = [None] * num_cameras
camera_thread = []

# Setup the cameras
for cam_id in range(num_cameras):
    cameras[cam_id] = devices.FaceCamera(cam_id)
    thread = CameraFaceDetector(cameras[cam_id], face_size)
    camera_thread.append(thread)
    thread.start()

# Neural Net Facial recognition start
facial_recognition_thread = FacialRecognition()
facial_recognition_thread.start()
#nn4_small2_pretrained = model.create_model()
#nn4_small2_pretrained.load_weights('nn/bin/nn4.small2.v1.h5')

# Start time
if fps_demo:
    start = time.time()
    num_frames = 0
    average_frames = 10
    fps = 0

# Main program
while True:

    for cam_id, camera in enumerate(cameras):

        frame = camera.getFrame()
        face = camera.getFace()

        if frame is not None:
            if face is not None:
                print("Face detected in camera " + str(cam_id))
                #cv2.imshow("Face " + str(cam_id), face)

                start = time.time()

                # Face Landmarks demo
                if face_detector_demo:
                    frame = demo.demo_face_detector(camera, frame)

                # Arduino hardware demo
                if serial_demo:
                    arduino.writeString("on")

                # Call the facial recognition thread with the face
                facial_recognition_thread.recognize_face(face)
                face_features = None

                # Wait for the face features to be computed
                while face_features is None:
                    face_features = facial_recognition_thread.get_face_features()

                # Get all persons from database
                persons = db.get_all_persons()
                threshold = 0.56
                match = "unknown"

                # Compare the distance with each person from DB
                for i, person in enumerate(persons):
                    distance = np.sum(np.square(face_features - np.fromstring(person.face_features, np.float32)))
                    if distance < threshold:
                        match = person.name
                        break

                # Arduino hardware demo
                if serial_demo:
                    arduino.writeString(match)

                # Wait to recognize next face - problems with imshow
                end = time.time()
                print("Recognized as: {0} in {1:.2f}s".format(match, end-start))
                print("")
                time.sleep(2)

            # Frames per second on screen
            if fps_demo:

                # Count the frames of every camera and avg the fps
                num_frames = float(num_frames + 1 / num_cameras)
                if num_frames > average_frames:
                    fps, end = demo.compute_fps(num_frames, start)
                    start = end
                    num_frames = 0
                frame = demo.demo_fps(camera, frame, fps)

            #cv2.imshow("Camera " + str(cam_id), cv2.resize(frame, tuple(int(x * camera.getScaleFactor() * 5) for x in camera.getDim())))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        for thread in camera_thread:
            thread.stop()
        facial_recognition_thread.stop()
        cv2.destroyAllWindows()
        break
