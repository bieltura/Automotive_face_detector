from face_detector.detector import CameraFaceDetector
from face_recognition.recognition import FacialRecognition


class Recognizer:
    def __init__(self, cam, stereo):

        # Face size (square in px for CNN)
        self.face_size = 240
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
        self.face_detector = CameraFaceDetector(cam.getScaleFactor(), self.face_size, stereo=stereo)
        self.face_detector.start()

    # Main program
    def recognize(self, stereo, frame_right, frame_left=None):

        if frame_right is not None:

            if not self.face_detected:

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

                # 3D model has been obtframe_right = ained
                else:
                    self.depth_detected = True

                    # Face and depth detected, recognize 3D:
                    self.facial_recognition_thread.recognize_face(self.face, self.face_3d)

            else:

                self.match = self.facial_recognition_thread.get_match()

                if self.match is not None:
                    print(self.match)

                # Restart the detector thread
                self.face_detected = False
                self.face_detector.detect(None, None)

                # Turn back to scan faces
                self.face = None
                self.face_landmarks = None

                if stereo:
                    self.face_3d = None
                    self.depth_detected = False

            if stereo:
                return self.face_landmarks, self.face, self.match, self.face_3d
            else:
                return self.face_landmarks, self.face, self.match

    def close(self):
        # Face detector thread stop
        self.face_detector.stop()
        self.facial_recognition_thread.stop()
