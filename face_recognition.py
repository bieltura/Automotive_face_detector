from threading import Thread, Lock
from nn import model
import numpy as np
import tensorflow as tf
from tensorflow import keras

class FacialRecognition(Thread):
    def __init__(self):
        Thread.__init__(self)

        self.face = None
        self.face_features = None

        self.nn4_small2_pretrained = None

        # Load the model
        # Load json and create model
        #json_file = open("nn/bin/nn4.small2.v1.json", 'r')
        #loaded_model_json = json_file.read()
        #json_file.close()
        #self.nn4_small2_pretrained = keras.models.model_from_json(loaded_model_json)

        print("Loading the model ...")
        self.nn4_small2_pretrained = model.create_model()
        self.nn4_small2_pretrained.load_weights('nn/bin/nn4.small2.v1.h5')

        # Tell the model is loaded in a different thread
        self.nn4_small2_pretrained._make_predict_function()
        print("Model loaded")

        # Variable to stop the camera thread if needed
        self.stopThread = False

    def run(self):

        while True:

            if not self.stopThread:

                # Lock the thread
                if self.face is not None:

                    # scale RGB values to interval [0,1]
                    face = (self.face / 255.).astype(np.float32)

                    self.nn4_small2_pretrained.load_weights('nn/bin/nn4.small2.v1.h5')

                    # Maybe we can align here?
                    self.face_features = self.nn4_small2_pretrained.predict(np.expand_dims(face, axis=0))[0]

                    self.face = None

            else:
                return

    def recognize_face(self, face):
        self.face = face

    def get_face_features(self):
        return self.face_features

    # State variable for stopping face detector service
    def stop(self):
        # Stop this face_detector thread
        self.stopThread = True
