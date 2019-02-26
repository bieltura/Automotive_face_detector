from utils import dlib_face_detecion as detector
from threading import Thread


# Detect face is frame
class CameraFaceDetector(Thread):

    def __init__(self, scale_factor, face_size):
        Thread.__init__(self)

        self.scale_factor = scale_factor
        self.face_size = face_size
        self.frame = None

        # Every face camera has a detector attribute
        self.detector = detector
        self.face = None
        self.landmarks = None

        # Variable to stop the camera thread if needed
        self.stopThread = False

    # Main thread method
    def run(self):

        while True:
            if not self.stopThread:

                if self.frame is None:
                    self.face = None
                else:
                    # Detect if there is a face in the frame
                    self.face, self.landmarks = self.detector.detect_face(self.frame, self.face_size, face_scale_factor=self.scale_factor)

            # End the thread and close the camera
            elif self.stopThread:
                return

    # State variable for stopping face detector service
    def stop(self):
        self.stopThread = True

    def getFace(self):
        return self.face

    def getLandmarks(self):
        return self.landmarks

    def detect(self, frame):
        self.frame = frame
