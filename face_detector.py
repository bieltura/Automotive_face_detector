from utils import dlib_face_detecion as detector
from threading import Thread

import time


# Runs captures until a face is detected
class CameraFaceDetector(Thread):

    def __init__(self, face_camera, face_size):
        Thread.__init__(self)

        self.face_camera = face_camera
        self.face_size = face_size

        # Every face camera has a detector attribute
        self.detector = detector

        # Variable to stop the camera thread if needed
        self.stopThread = False
        self.pauseThread = False

    # Main thread method
    def run(self):

        # Assume no face detected when camera is first up
        self.face_camera.setFace(None)

        while True:
            if not self.stopThread and not self.pauseThread:

                # Get the frame
                frame = self.face_camera.getFrame()

                if frame is not None:
                    # Detect if there is a face in the frame
                    face, landmarks = self.detector.detect_face(frame, self.face_size, face_scale_factor=self.face_camera.getScaleFactor())

                    # Set the face of the detected face (with the dimensions specified) - even if it is None
                    self.face_camera.setFace(face, landmarks=landmarks)

            # End the thread and close the camera
            elif self.stopThread:
                return

    # State variable for stopping face detector service
    def stop(self):
        # Stop this face_detector thread
        self.stopThread = True

    def pause(self):
        self.pauseThread = True

    def resume(self):
        self.pauseThread = False


