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
            self.scene_3d = None

        # Every face camera has a detector attribute
        self.detector = detector
        self.face = None

        # Face attributes
        self.landmarks = None
        self.bb = None

        # Variable to stop the camera thread if needed
        self.stopThread = False

    # Main thread method
    def run(self):

        while True:
            if not self.stopThread:

                # If there is no frame, no face detected
                if self.frame is None:
                    self.face = None
                    if self.stereo:
                        self.face_3d = None
                        self.scene_3d = None
                else:

                    # 3D recognition:
                    if self.stereo:

                        # Check if a face appears in the secundary frame (depth map will be optained with the main)
                        self.face, self.landmarks, self.bb = self.detector.detect_face(self.second_frame, self.face_size, face_scale_factor=self.scale_factor)

                        # A face has been detected, create the depth map
                        if self.face is not None:
                            self.face_3d, self.scene_3d = detector_3d.detect_3d_face(self.frame, self.second_frame, ROI=self.bb)

                    # 2D recognition
                    else:
                        self.face, self.landmarks, self.bb = self.detector.detect_face(self.frame, self.face_size, face_scale_factor=self.scale_factor)

            # End the thread and close the camera
            elif self.stopThread:
                return

    def stop(self):
        self.stopThread = True

    def getFace(self):
        face = self.face
        self.face = None
        return face

    def get3dFace(self):
        face_3d = self.face_3d
        self.face_3d = None
        return face_3d, self.scene_3d

    def getFaceAtributtes(self):
        return self.landmarks, self.bb

    def detect(self, frame, second_frame=None):
        if self.face is None:
            self.frame = frame
            if self.stereo:
                self.second_frame = second_frame
