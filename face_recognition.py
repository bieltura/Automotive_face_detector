from threading import Thread
import time


class FacialRecognition(Thread):
    def __init__(self, face):
        Thread.__init__(self)

        self.face = face

        # Variable to stop the camera thread if needed
        self.stopThread = False

    def run(self):
        time.sleep(2)