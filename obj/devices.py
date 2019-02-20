import cv2


class Camera:

    def __init__(self, cam_id, focal_length=4.46):
        self.id = cam_id
        self.capture = None
        self.frame = None

        # Open the device to get properties of the cam
        self.open()

        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.sensor = {
            "focal_length": focal_length,
            "sensor_type": "1/2.7",
            "sensor_height": 5.37
        }

    def open(self):
        self.capture = cv2.VideoCapture(self.id)

    def captureFrame(self):
        if self.capture.isOpened():
            ret, self.frame = self.capture.read()
        else:
            self.frame = None

        return self.getFrame()

    def getFrame(self):
        return self.frame

    def getSensorInfo(self):
        return self.sensor

    def close(self):
        self.capture.release()

    def getDim(self):
        return self.width, self.height


class FaceCamera(Camera):

    def __init__(self, cam_id, min_face_size=40, max_face_dist=750):
        super().__init__(cam_id)
        self.face = None
        self.landmarks = None

        # Compute scale factor for different focal length: Average human face height is 55cm
        face_px_height = self.sensor["focal_length"] * 400 * self.height / (max_face_dist * self.sensor["sensor_height"])
        self.scale_factor = min_face_size / face_px_height
        print(face_px_height/min_face_size)

    def getFace(self):
        return self.face

    def setFace(self, face, landmarks=None):
        self.face = face
        self.landmarks = landmarks

    def getScaleFactor(self):
        return self.scale_factor

    def getLandmarks(self):
        return self.landmarks
