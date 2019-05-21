from face_detector import dlib_face_detecion as detector
from stereo_vision import stereo as detector_3d
from threading import Thread
from utils import demo


# Detect face is frame
class CameraFaceDetector(Thread):

    def __init__(self, scale_factor, face_size, stereo=False):
        Thread.__init__(self)

        self.scale_factor = scale_factor
        self.face_size = face_size
        self.stereo = stereo
        self.frame = None

        # Every face camera has a detector attribute
        self.alignedFace = None
        self.landmarkFace = None

        # Stereo properties
        if self.stereo:
            self.second_frame = None
            self.face_3d = None
            self.scene_3d = None

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
                    self.alignedFace = None
                    self.landmarkFace = None
                    if self.stereo:
                        self.face_3d = None
                        self.scene_3d = None
                else:
                    # Make a copy of this frame to analyze
                    frame = self.frame.copy()

                    # 3D recognition:
                    if self.stereo:

                        if self.second_frame is not None:
                            # Make a copy of the second frame to analyze
                            second_frame = self.second_frame.copy()

                            # Check if a face appears in the secundary frame (depth map will be optained with the main)
                            self.alignedFace, self.landmarks, self.bb = detector.detect_face(second_frame.copy(), self.face_size, face_scale_factor=self.scale_factor)

                            # A face has been detected, create the depth map and landmarks
                            if self.alignedFace is not None:
                                self.landmarkFace = demo.landmarks_img(second_frame.copy(), self.landmarks, self.bb, self.face_size)
                                self.face_3d, self.scene_3d = detector_3d.detect_3d_face(frame.copy(), second_frame.copy(), self.face_size, ROI=self.bb)

                    # 2D recognition
                    else:
                        self.alignedFace, self.landmarks, self.bb = detector.detect_face(frame.copy(), self.face_size, face_scale_factor=self.scale_factor)

                        if self.alignedFace is not None:
                            self.landmarkFace = demo.landmarks_img(frame.copy(), self.landmarks, self.bb, self.face_size)

            # End the thread and close the camera
            elif self.stopThread:
                return

    def stop(self):
        self.stopThread = True

    def getFace(self):
        face = self.alignedFace
        landmarkFace = self.landmarkFace
        self.alignedFace = None
        self.landmarkFace = None
        return face, landmarkFace

    def get3dFace(self):
        face_3d = self.face_3d
        self.face_3d = None
        return face_3d, self.scene_3d

    def getFaceAtributtes(self):
        return self.landmarks, self.bb

    def detect(self, frame, second_frame=None):
        if self.alignedFace is None:
            self.frame = frame
            if self.stereo:
                self.second_frame = second_frame
