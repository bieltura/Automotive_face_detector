from utils import dlib_face_detecion as detector
from threading import Thread


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

    # Main thread method
    def run(self):

        # Assume no face detected when camera is first up
        self.face_camera.setFace(None)

        while True:
            if not self.stopThread:

                # Get the frame
                frame = self.face_camera.captureFrame()

                if frame is not None:
                    # Detect if there is a face in the frame
                    face, landmarks = self.detector.detect_face(frame, self.face_size)

                    # Set the face of the detected face (with the dimensions specified) - even if it is None
                    self.face_camera.setFace(face, landmarks=landmarks)

            # End the thread and close the camera
            else:
                self.face_camera.close()
                return

    # State variable for stopping face detector service
    def stop(self):
        # Stop this face_detector thread
        self.stopThread = True

