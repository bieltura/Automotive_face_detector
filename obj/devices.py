import cv2
import numpy as np


class Camera:

    def __init__(self, cam_id):
        self.id = cam_id
        self.capture = None
        self.frame = None
        self.open()

        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def open(self):
        self.capture = cv2.VideoCapture(self.id)

    def captureFrame(self):
        self.frame = None

        if self.capture.isOpened():
            ret, self.frame = self.capture.read()

        return self.getFrame()

    def getFrame(self):
        return self.frame

    def close(self):
        self.capture.release()

    def getDim(self):
        return self.width, self.height


class FaceCamera(Camera):

    def __init__(self, cam_id):
        super().__init__(cam_id)
        self.face = None

    def getFace(self):
        return self.face

    def setFace(self, face):
        self.face = face
