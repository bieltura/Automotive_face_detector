from threading import Thread, Lock
from face_recognition import model
import numpy as np
from database import db_service as db
import tensorflow as tf
from tensorflow import keras

class FacialRecognition(Thread):
    def __init__(self):
        Thread.__init__(self)

        self.face = None
        self.face_features = None
        self.match = None

        self.nn4_small2_pretrained = None

        # Load the model
        # Load json and create model
        #json_file = open("nn/bin/nn4.small2.v1.json", 'r')
        #loaded_model_json = json_file.read()
        #json_file.close()
        #self.nn4_small2_pretrained = keras.models.model_from_json(loaded_model_json)

        print("Loading the facial recognition model ...")
        self.nn4_small2_pretrained = model.create_model()
        self.nn4_small2_pretrained.load_weights('face_recognition/bin/nn4.small2.v1.h5')

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

                    # Reset the values, make sure we lock the enter for more faces
                    self.face = None

                    self.nn4_small2_pretrained.load_weights('face_recognition/bin/nn4.small2.v1.h5')

                    # Forward pass NN
                    self.face_features = self.nn4_small2_pretrained.predict(np.expand_dims(face, axis=0))[0]

                    if self.face_features is not None:

                        # Get all persons from database
                        persons = db.get_all_persons()
                        threshold = 0.56
                        self.match = "unknown"

                        # Compare the distance with each person from DB
                        for i, person in enumerate(persons):
                            distance = np.sum(
                                np.square(self.face_features - np.fromstring(person.face_features, np.float32)))
                            if distance < threshold:
                                self.match = person.name
                                break

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
