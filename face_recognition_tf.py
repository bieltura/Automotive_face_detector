from threading import Thread, Lock
import tensorflow as tf
from tensorflow import keras

GRAPH_PB_PATH = 'nn/20180408-102900/20180408-102900.pb'

class FacialRecognition(Thread):
    def __init__(self):
        Thread.__init__(self)

        self.face = None
        self.face_features = None

        self.nn4_small2_pretrained = None

        # Load the model
        print("Loading the model ...")
        with tf.gfile.GFile(GRAPH_PB_PATH, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())


        # graph_def to the current default graph:
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="facenet")
            self.graph = graph

        print("Model Loaded")

        for op in self.graph.get_operations():
            print(op.name)

        # Variable to stop the camera thread if needed
        self.stopThread = False

    def run(self):
        print("Model loaded")

        while True:

            if not self.stopThread:

                # Lock the thread
                if self.face is not None:

                    # Acces the input and output nodes
                    x = self.graph.get_tensor_by_name("facenet/input:0")
                    print(x)

                    y = self.graph.get_tensor_by_name("facenet/embeddings:0")
                    print(y)

                    # Launch a session: TODO
                    with tf.Session(graph=self.graph) as sess:
                        face_features = sess.run(y, feed_dict={x: self.face})

                    print(face_features)

                    # scale RGB values to interval [0,1]
                    #face = (self.face / 255.).astype(np.float32)

                    #self.nn4_small2_pretrained.load_weights('nn/bin/nn4.small2.v1.h5')

                    # Maybe we can align here?
                    #self.face_features = self.nn4_small2_pretrained.predict(np.expand_dims(face, axis=0))[0]

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
