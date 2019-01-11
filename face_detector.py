from utils import haar_face_detection as fd
from threading import Thread


class CameraFaceDetector(Thread):
    def __init__(self, face_camera, face_size):
        Thread.__init__(self)
        self.face_camera = face_camera
        self.face_size = face_size
        self.stopThread = False
        pass

    def run(self):
        self.face_camera.setFace(None)

        while True:
            if not self.stopThread:

                frame = self.face_camera.captureFrame()

                if frame is not None:

                    self.face_camera.setFace(fd.detect_face(frame, self.face_size, self.face_size))

                    if self.face_camera.getFace() is not None:
                        self.face_camera.close()
                        return

            # Close the program
            else:
                self.face_camera.close()
                return

    def stop(self):
        self.stopThread = True
