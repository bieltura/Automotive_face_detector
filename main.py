import cv2
import glob
import time
from obj import devices
from face_detector import CameraFaceDetector
from face_recognition import FacialRecognition
from utils import demo
from utils.port_serial import arduino_serial

# Demonstration values:
face_detector_demo = False
face_recognition_demo = False
fps_demo = False
serial_demo = False

# Check number of cameras in Linux distribution cameras
num_cameras = 0
for camera in glob.glob("/dev/video?"):
    num_cameras += 1

# Check number of port serials in Linux distribution
if serial_demo:
    for port, camera in enumerate(glob.glob("/dev/ttyUSB?")):
        arduino = arduino_serial('/dev/ttyUSB'+str(port))

# Face size (square in px for CNN)
face_size = 96

# Number of cameras in the system
cameras = [None] * num_cameras
face_detector = [None] * num_cameras
face_detected = [False] * num_cameras

# Currently one face_features for n cameras
face_features = None

# Neural Net Facial recognition start
facial_recognition_thread = FacialRecognition()
facial_recognition_thread.start()

# Setup the cameras
for cam_id in range(num_cameras):
    cameras[cam_id] = devices.FaceCamera(cam_id)
    face_detector[cam_id] = CameraFaceDetector(cameras[cam_id].getScaleFactor(), face_size)
    cameras[cam_id].start()
    face_detector[cam_id].start()

# Start time
if fps_demo:
    start = time.time()
    num_frames = 0
    average_frames = 10
    fps = 0

# Main program
while True:

    for cam_id, camera in enumerate(cameras):

        # Get the frame from the camera
        frame = camera.getFrame()

        # If we have not detected any face
        if not face_detected[cam_id]:

            if frame is not None:
                face_detector[cam_id].detect(frame)
                face = face_detector[cam_id].getFace()

                if face is not None:
                    print("Face detected in camera " + str(cam_id))

                    # Arduino hardware demo
                    if serial_demo:
                        arduino.writeString("Detecting ...")

                    start = time.time()

                    # Pause the face detector thread by setting a Nonr frame
                    face_detector[cam_id].detect(None)
                    face_detected[cam_id] = True
                    cv2.imshow("Face " + str(cam_id), face)

                    # Call the facial recognition thread with the face
                    facial_recognition_thread.recognize_face(face)

                    # Face Landmarks demo
                    if face_detector_demo:
                        frame = demo.demo_face_detector(camera, frame)

        # If the face has been detected check the face features
        if face_detected[cam_id]:

            match = facial_recognition_thread.get_match()

            # If they are computed
            if match:

                # Arduino hardware demo
                if serial_demo:
                    arduino.writeString(match)

                # Wait to recognize next face - problems with imshow
                end = time.time()
                print("Recognized as: {0} in {1:.2f}s".format(match, end - start))
                print("")

                # Once recognized, resume the face detector
                face_detected[cam_id] = False

        # Frames per second on screen
        if fps_demo:

            # Count the frames of every camera and avg the fps
            num_frames = float(num_frames + 1 / num_cameras)
            if num_frames > average_frames:
                fps, end = demo.compute_fps(num_frames, start)
                start = end
                num_frames = 0
            frame = demo.demo_fps(camera, frame, fps)

        if frame is not None:
            cv2.imshow("Camera " + str(cam_id),
               cv2.resize(frame, tuple(int(x * camera.getScaleFactor() * 5) for x in camera.getDim())))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Face detector thread stop
        for thread in face_detector:
            thread.stop()

        # Camera thread stop
        for camera in cameras:
            camera.close()

        # Facial recognition Thread stop
        facial_recognition_thread.stop()

        cv2.destroyAllWindows()
        break
