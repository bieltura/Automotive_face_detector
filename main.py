from face_detector.detector import CameraFaceDetector
from face_recognition.recognition import FacialRecognition

import cv2

num_face = 990

num_face = 799


class Recognizer:
    def __init__(self, scaleFactor, stereo):

        # Face size (square in px for CNN)
        self.face_size = 96
        self.face = None
        self.face_landmarks = None

        self.face_detected = False
        self.match = None

        if stereo:
            self.depth_detected = False
            self.face_3d = None

        # Neural Net Facial recognition start
        self.facial_recognition_thread = FacialRecognition(stereo=stereo)
        self.facial_recognition_thread.start()

        # Any of the two cameras for the scalefactor
        self.face_detector = CameraFaceDetector(scaleFactor, self.face_size, stereo=stereo)
        self.face_detector.start()

    # Main program
    def recognize(self, stereo, frame_right, frame_left=None):

        global num_face

        if frame_right is not None:

            if not self.face_detected:

                # Not face detected, restore the match and the 3D
                self.match = None
                self.face_3d = None

                # Pass the frames to detect the face, generates the depth map and the face
                self.face_detector.detect(frame_right.copy(), frame_left)

                # If there is no face, get the face from the detector
                if self.face is None:

                    # Get the aligned face and landmark face
                    self.face, self.face_landmarks = self.face_detector.getFace()

                # Face has been detected
                else:
                    self.face_detected = True

                    if stereo:
                        # Retrieve the depth
                        self.depth_detected = False

                    else:
                        # Face detected, recognize 2D:
                        self.facial_recognition_thread.recognize_face(self.face.copy())

            elif stereo and not self.depth_detected:
                # Once the face is detected, get the 3D model from the stereo
                if self.face_3d is None:
                    self.face_3d, scene = self.face_detector.get3dFace()
                    #cv2.imshow('scene', scene)

                # 3D model has been obtained
                else:
                    self.depth_detected = True

                    #cv2.imwrite('paper/3d_paper' + str(num_face) + '.png', self.face_3d)
                    #cv2.imwrite('real_b/aligned' + str(num_face) + '.png', self.face)
                    #num_face = num_face + 1

                    # Face and depth detected, recognize 3D:
                    self.facial_recognition_thread.recognize_face(self.face, self.face_3d)

            else:

                self.match = self.facial_recognition_thread.get_match()

                # Restart the detector thread
                self.face_detected = False
                self.face_detector.detect(None, None)

                # Turn back to scan faces
                self.face = None
                self.face_landmarks = None

                if stereo:
                    #self.face_3d = None
                    self.depth_detected = False

            if stereo:
                return self.face_landmarks, self.face, self.match, self.face_3d
            else:
                return self.face_landmarks, self.face, self.match

    def close(self, stereo):
        # Face detector thread stop
        self.face_detector.stop()
        self.facial_recognition_thread.stop()

        self.face = None
        self.face_landmarks = None

        self.face_detected = False
        self.match = None

        if stereo:
            self.depth_detected = False
            self.face_3d = None