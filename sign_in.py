from database import db_service as db
import cv2
import numpy as np
from obj import devices
from face_recognition import FacialRecognition
from utils.dlib_face_alignment import AlignDlib
import dlib

# Scale factor! Adaptive to the dimensions!!!
face_scale_factor = 1/15
alignment = AlignDlib('data/dlib/shape_predictor_68_face_landmarks.dat')

# Open the camera to capture the picture
camera = devices.FaceCamera(0)

# Open the database
db.create_database()

# Neural Net Facial recognition start
facial_recognition_thread = FacialRecognition()

# Calls to the NN HERE
print("Start thread for NN")
facial_recognition_thread.start()


while True:
    camera.captureFrame()
    frame = camera.getFrame()

    cv2.imshow("Log in camera", cv2.resize(frame, tuple(int(x / 2) for x in camera.getDim())))

    # Save the image
    if cv2.waitKey(1) & 0xFF == ord('s'):
        # Get the name of the person
        name = input("Name: ")

        # Use the name to generate the picture
        picture_path = "database/img/" + name.replace(" ", "_") + ".jpg"
        cv2.imwrite(picture_path, frame)

        # Detect the bounding box
        bb = alignment.getLargestFaceBoundingBox(frame)

        # Crop the image from the boundary box (the big)
        face_aligned = alignment.align(96, frame, bb, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

        facial_recognition_thread.recognize_face(face_aligned)

        face_features = None

        print("Calculating face features...")

        # Wait for the face features to be computed
        while face_features is None:
            face_features = facial_recognition_thread.get_face_features()


        # Create the person in the database
        db.add_person(name, picture_path, face_features)

        # Print the person registered:
        person = db.get_persons_by_name(name)

        print(" ")
        print("PERSON ID: " + str(person.id))
        print("NAME: " + str(person.name))
        print("PICTURE: " + str(person.face_img_path))
        print("FACE FEATURES: ")


        print(np.fromstring(person.face_features, np.float32))

        facial_recognition_thread.stop()

        cv2.destroyAllWindows()
        break

    # Quit the program with q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
