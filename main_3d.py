import cv2
import glob
from obj import devices
from face_detector import CameraFaceDetector
import numpy as np
from face_recognition import FacialRecognition

# Demonstration values:
face_detector_demo = False
face_recognition_demo = False

# Check number of cameras in Linux distribution cameras
num_cameras = 0
for camera in glob.glob("/dev/video?"):
    num_cameras += 1

# Face size (square in px for CNN)
face_size = 96
face = None
face_3d = None

# Number of cameras in the system
cameras = [None] * num_cameras
face_detected = False

# Currently one face_features for n cameras
face_features = None

# Neural Net Facial recognition start
#facial_recognition_thread = FacialRecognition()
#facial_recognition_thread.start()

# Setup the cameras
for cam_id in range(num_cameras):
    cameras[cam_id] = devices.FaceCamera(cam_id)
    cameras[cam_id].start()

face_detector = CameraFaceDetector(cameras[cam_id].getScaleFactor(), face_size, stereo=True)
face_detector.start()

# Main program
while True:

    # Get the frame from the camera
    frame_right = cameras[0].getFrame()
    frame_left = cameras[1].getFrame()

    if frame_right is not None and frame_left is not None:

        if not face_detected:

            # Pass the frames to detect the face
            face_detector.detect(frame_right, frame_left)

            # If there is no face, get the face from the detector
            if face is None:
                face = face_detector.getFace()

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

                # Stop the face detector thread:
                face_detected = False

                # Txt saving for dataset
                #np.savetxt('faces/face_paper{}_3d.txt'.format(num_face), face_3d)
                #num_face = num_face + 1

                cv2.imshow("Face " + str(cam_id), np.uint8(np.interp(face_3d, (face_3d.min(), face_3d.max()), (0, 255))))
                cv2.imshow("Face no scale " + str(cam_id), scene)

                # Turn back to scan faces
                face_3d = None
                face = None
                face_detected = False

                print("3D Face model done in camera " + str(cam_id))

        cv2.imshow("Camera " + str(cam_id), cv2.resize(frame_right, tuple(int(x * cameras[0].getScaleFactor() * 5) for x in cameras[0].getDim())))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Face detector thread stop
        face_detector.stop()

        # Camera thread stop
        for camera in cameras:
            camera.close()

        cv2.destroyAllWindows()
        break
