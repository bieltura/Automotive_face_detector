import cv2
from devices import cameras
from face_detector.detector import CameraFaceDetector
from face_recognition.recognition import FacialRecognition
import numpy as np

# 2 cameras for the stereo system
num_cameras = 2
cam = [None] * num_cameras

# Face size (square in px for CNN)
face_size = 240
face = None
face_3d = None
face_detected = False
depth_detected = False

num_face = 527

# Neural Net Facial recognition start
facial_recognition_thread = FacialRecognition(stereo=True)
facial_recognition_thread.start()


# Setup the cameras
for cam_id in range(num_cameras):
    cam[cam_id] = cameras.FaceCamera(cam_id)
    cam[cam_id].start()

# Any of the two cameras for the scalefactor
face_detector = CameraFaceDetector(cam[cam_id].getScaleFactor(), face_size, stereo=True)
face_detector.start()

# Main program
while True:

    # Get the frame from the camera
    frame_right = cam[0].getFrame()
    frame_left = cam[1].getFrame()

    if frame_right is not None and frame_left is not None:

        if not face_detected:

            # Pass the frames to detect the face, generates the depth map and the face
            face_detector.detect(frame_right, frame_left)

            # If there is no face, get the face from the detector
            if face is None:

                # Get the aligned face and landmark face
                face, face_landmarks = face_detector.getFace()

            # Face has been detected
            else:
                face_detected = True

                # Retrieve the depth
                depth_detected = False

                #print("Face detected in camera " + str(cam_id))

        elif not depth_detected:
            # Once the face is detected, get the 3D model from the stereo
            if face_3d is None:
                face_3d, scene = face_detector.get3dFace()

            # 3D model has been obtained
            else:
                depth_detected = True

                # Face and depth detected, recognize:
                facial_recognition_thread.recognize_face(face, face_3d)

        else:

            #num_face = num_face + 1
            match = facial_recognition_thread.get_match()

            if match is not None:
                print(match)

            if face is not None:
                cv2.imshow("Aligned face", np.uint8(face))

            if face_landmarks is not None:
                cv2.imshow("68 Landmarks", np.uint8(face_landmarks))
                #cv2.imwrite("face.png", face)
                #cv2.imwrite("face_land.png", face_landmarks)
                #cv2.imwrite("full.png", frame_left)

            if face_3d is not None:
                cv2.imshow("Face depth map", np.uint8(face_3d))
                #cv2.imshow("Scene depth map", np.uint8(scene))
                #cv2.imshow("3D Scene " + str(cam_id), np.uint8(scene))

            # Save the values

            #cv2.imwrite("no_faces/3d_paper"+str(num_face)+".png",face_3d)
            #cv2.imwrite("faces/3d_scene" + str(num_face) + ".png", scene)
            #cv2.imwrite("faces/2d_scene" + str(num_face) + ".png", frame_aux)

            # Turn back to scan faces
            face = None
            face_landmarks = None
            face_3d = None

            # Restart the detector thread
            face_detected = False
            face_detector.detect(None, None)

        cv2.imshow("Right camera" + str(cam_id), cv2.resize(frame_right, tuple(int(x * cam[0].getScaleFactor() * 2) for x in cam[0].getDim())))
        cv2.imshow("Left camera" + str(cam_id), cv2.resize(frame_left, tuple(int(x * cam[0].getScaleFactor() * 2) for x in cam[0].getDim())))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Face detector thread stop
        face_detector.stop()
        facial_recognition_thread.stop()

        # Camera thread stop
        for camera in cam:
            camera.close()

        cv2.destroyAllWindows()
        break
