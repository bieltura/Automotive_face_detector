import cv2


class Camera:

    def __init__(self, cam_id):
        self.id = cam_id
        self.capture = None
        self.open()

    def open(self):
        self.capture = cv2.VideoCapture(self.id)

    def getFrame(self):
        frame = None

        if self.capture.isOpened():
            ret, frame = self.capture.read()

        return frame

    def close(self):
        self.capture.release()
