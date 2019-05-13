from threading import Thread, Lock
from face_recognition.depth import model_3d
import numpy as np
from database import db_service as db
import tensorflow as tf

class depth_rec(Thread):
    def __init__(self):
        Thread.__init__(self)

        self.face = None
        self.is_person = False

        print("Loading the facial 3D recognition model ...")
        self.depth_net = model_3d.create_model("face_recognition/depth/binary_depth_classification.h5", (240,240,1))

        self.depth_net._make_predict_function()

        # Tell the model is loaded in a different thread
        print("Model loaded")

        # Variable to stop the camera thread if needed
        self.stopThread = False

        self.session = tf.Session()

    def run(self):

        while True:

            if not self.stopThread:

                # Lock the thread
                if self.face is not None:

                    # scale RGB values to interval [0,1]
                    face = (self.face / 255.).astype(np.float32)

                    # Reset the values, make sure we lock the enter for more faces
                    self.face = None

                    self.nn4_small2_pretrained = model_3d.create_model(
                        "face_recognition/depth/binary_depth_classification.h5", (240, 240, 1))

                    # Forward pass NN
                    self.face_features = self.nn4_small2_pretrained.predict(np.expand_dims(np.expand_dims(face, axis=0),axis=4))

                    print(self.face_features)

            else:
                return

    def recognize_face(self, face):
        self.face = face

    # State variable for stopping face detector service
    def stop(self):
        self.stopThread = True

    def get_match(self):
        match = self.match
        self.match = None
        return match
