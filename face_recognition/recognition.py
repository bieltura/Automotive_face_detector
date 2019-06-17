from threading import Thread
from face_recognition.faceNet import model as facenet_model
from face_recognition.depth import model as depth_model
import numpy as np
from database import db_service as db
import tensorflow as tf

class FacialRecognition(Thread):
    def __init__(self, stereo=False):
        Thread.__init__(self)

        self.face = None
        self.depth_face = None
        self.face_features = None

        self.nn4_small2_pretrained = None
        self.nn_binary_depth_map = None

        # Create the session
        self.thread_session = tf.Session()

        # Create the model
        print("Loading the FaceNet recognition model ...")
        self.nn4_small2_pretrained = facenet_model.create_model('face_recognition/faceNet/bin/nn4.small2.v1.h5')

        # Tell the model is loaded in a different thread
        self.nn4_small2_pretrained._make_predict_function()

        # FaceNet graph
        self.facenet_graph = tf.get_default_graph()

        if stereo:
            print("Loading the Depth detection model ...")
            self.nn_depth = depth_model.create_model('face_recognition/depth/bin/binary_depth_classification.h5')
            self.nn_depth._make_predict_function()

            self.depth_graph = tf.get_default_graph()
        else:
            self.nn_depth = None

        print("Model loaded")

        # Variable to stop the camera thread if needed
        self.stopThread = False

    def run(self):

        while True:

            if not self.stopThread:

                # Remove the face detected and make a copy to pass to the NN
                if self.face is not None:
                    face = (self.face / 255.).astype(np.float32)
                    self.face = None

                    # Remove the depth mapp and make a copy to pass to the NN
                    if self.depth_face is not None:
                        depth_face = (self.depth_face / 255.).astype(np.float32)
                        self.depth_face = None

                    # With the tensorflow session (new keras)
                    with self.thread_session.as_default():

                        depth_info = 0

                        # If there is a depth neural net
                        if self.nn_depth is not None:

                            # With the depth model as the default graph:
                            with self.depth_graph.as_default():

                                # Keras bug_ reload the weights for every iteration of the thread
                                self.nn_depth.load_weights('face_recognition/depth/bin/binary_depth_classification.h5')

                                depth_info = self.nn_depth.predict(np.expand_dims(np.expand_dims(depth_face, axis=0),axis=3))[0]

                                print("Depth confidence: {}".format(depth_info))

                        # If the detected depth map is a face or is 2D scanning
                        if depth_info > 0.4 or self.nn_depth is None:

                            with self.facenet_graph.as_default():

                                # Keras bug: reload the weights for every iteration of the thread
                                self.nn4_small2_pretrained.load_weights('face_recognition/faceNet/bin/nn4.small2.v1.h5')

                                self.face_features = self.nn4_small2_pretrained.predict(np.expand_dims(face, axis=0))[0]

                        else:
                            self.face_features = None

            else:
                return

    def recognize_face(self, face, depth_face=None):
        self.face = face
        self.depth_face = depth_face

    def get_face_features(self):
        return self.face_features

    # State variable for stopping face detector service
    def stop(self):
        self.stopThread = True

    def get_match(self):
        match = "unknown"
        if self.face_features is not None:

            # Get all persons from database
            persons = db.get_all_persons()
            distance = [1] * len(persons)

            # Compare the distance with each person from DB
            for i, person in enumerate(persons):
                distance[i] = np.sum(np.square(self.face_features - np.fromstring(person.face_features, np.float32)))

            if distance is not None:
                if min(distance) < 0.45:
                    match = persons[distance.index(min(distance))].name

            self.face_features = None

        return match
