from utils import dlib_face_detecion as detector
from stereo import stereo_vision as detector_3d
from threading import Thread


# Detect face is frame
class CameraFaceDetector(Thread):

    def __init__(self, scale_factor, face_size, stereo=False):
        Thread.__init__(self)

        self.scale_factor = scale_factor
        self.face_size = face_size
        self.stereo = stereo
        self.frame = None
        if self.stereo:
            self.second_frame = None
            self.detector_3d = detector_3d
            self.face_3d = None

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
                    face, self.landmarks, bb = self.detector.detect_face(self.frame, self.face_size, face_scale_factor=self.scale_factor)

                    # 3D recognition:
                    if self.stereo:
                        if self.second_frame is not None:
                            self.face_3d = detector_3d.detect_3d_face(self.frame, self.second_frame, ROI=bb, face_scale_factor=self.scale_factor)
                            self.face = face
                        else:
                            self.face = None
                    else:
                        self.face = face

            # End the thread and close the camera
            elif self.stopThread:
                return

    # State variable for stopping face detector service
    def stop(self):
        self.stopThread = True

    def getFace(self):
        return self.face

    def get3dFace(self):
        return self.face_3d

    def getLandmarks(self):
        return self.landmarks

    def detect(self, frame, second_frame=None):
        self.frame = frame
        if self.stereo:
            self.second_frame = second_frame
