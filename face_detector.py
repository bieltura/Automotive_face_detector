from utils import haar_face_detection as fd
from threading import Thread


# Runs captures until a face is detected
class CameraFaceDetector(Thread):

    def __init__(self, face_camera, face_size):
        Thread.__init__(self)

        self.face_camera = face_camera
        self.face_size = face_size

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

                    # Detect a face within that frame captured
                    self.face_camera.setFace(fd.detect_face(frame, self.face_size, self.face_size))

                    # Face detected, close the camera service to analyze the face
                    if self.face_camera.getFace() is not None:
                        return

            # End the thread and close the camera
            else:
                self.face_camera.close()
                return

    # State variable for stopping face detector service
    def stop(self):
        self.stopThread = True
