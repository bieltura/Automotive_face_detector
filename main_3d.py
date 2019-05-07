import cv2
import glob
from devices import cameras
from face_detector.detector import CameraFaceDetector
import numpy as np
from face_recognition.recognition import FacialRecognition
from utils import demo

# Demonstration values:
face_detector_demo = False
face_recognition_demo = False

# Check number of cameras in Linux distribution cameras
num_cameras = 0
for camera in glob.glob("/dev/video?"):
    num_cameras += 1

# Face size (square in px for CNN)
face_size = 120
face = None
face_3d = None
face_detected = False

num_face = 270

# Number of cameras in the system
cam = [None] * num_cameras

# Currently one face_features for n cameras
face_features = None

# Neural Net Facial recognition start
#facial_recognition_thread = FacialRecognition()
#facial_recognition_thread.start()

# Setup the cameras
for cam_id in range(num_cameras):
    cam[cam_id] = cameras.FaceCamera(cam_id)
    cam[cam_id].start()

face_detector = CameraFaceDetector(cam[cam_id].getScaleFactor(), face_size, stereo=True)
face_detector.start()

# Main program
while True:

    # Get the frame from the camera
    frame_right = cam[0].getFrame()
    frame_left = cam[1].getFrame()

    if frame_right is not None and frame_left is not None:

        if not face_detected:

            # Pass the frames to detect the face
            face_detector.detect(frame_right, frame_left)

            # If there is no face, get the face from the detector
            if face is None:

                # Get the aligned FACE
                face, face_landmarks = face_detector.getFace()

            # Face has been detected
            else:
                face_detected = True
                print("Face detected in camera " + str(cam_id))

        else:
            # Once the face is detected, get the 3D model from the stereo
            if face_3d is None:
                face_3d, scene = face_detector.get3dFace()

            # 3D model has been obtained
            else:

                #num_face = num_face + 1

                if face is not None:
                    cv2.imshow("Aligned face", np.uint8(face))

                if face_landmarks is not None:
                    cv2.imshow("68 Landmarks", np.uint8(face_landmarks))
                    #cv2.imwrite("face.png", frame_left)
                    #cv2.imwrite("face_land.png", face_landmarks)

                if face_3d is not None:
                    cv2.imshow("Face depth map", np.uint8(face_3d))
                    #cv2.imshow("3D Scene " + str(cam_id), np.uint8(scene))

                # Save the values

                #cv2.imwrite("faces/3d_face"+str(num_face)+".png",face_3d)
                #cv2.imwrite("faces/3d_scene" + str(num_face) + ".png", scene)
                #cv2.imwrite("faces/2d_scene" + str(num_face) + ".png", frame_aux)

                # Turn back to scan faces
                face = None
                face_landmarks = None
                face_3d = None

                # Restart the detector thread
                face_detected = False
                face_detector.detect(None, None)

        cv2.imshow("Right camera" + str(cam_id), cv2.resize(frame_right, tuple(int(x * cam[0].getScaleFactor() * 3) for x in cam[0].getDim())))
        cv2.imshow("Left camera" + str(cam_id), cv2.resize(frame_left, tuple(int(x * cam[0].getScaleFactor() * 3) for x in cam[0].getDim())))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Face detector thread stop
        face_detector.stop()

        # Camera thread stop
        for camera in cam:
            camera.close()

        cv2.destroyAllWindows()
        break
