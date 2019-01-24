from database import db_service as db
import cv2
import numpy as np
from obj import devices

# Open the camera to capture the picture
camera = devices.FaceCamera(0)

# Open the database
db.create_database()


while True:
    frame = camera.captureFrame()

    cv2.imshow("Log in camera", cv2.resize(camera.getFrame(), tuple(int(x / 2) for x in camera.getDim())))

    # Save the image
    if cv2.waitKey(1) & 0xFF == ord('s'):
        # Get the name of the person
        name = input("Name: ")

        # Use the name to generate the picture
        picture_path = "database/img/" + name.replace(" ", "_") + ".jpg"
        cv2.imwrite(picture_path, camera.getFrame())

        # Create the person in the database
        db.add_person(name, picture_path)

        # Print the person registered:
        person = db.get_persons_by_name(name)

        print(" ")
        print("PERSON ID: " + str(person.id))
        print("NAME: " + str(person.name))
        print("PICTURE: " + str(person.face_img_path))
        print("FACE FEATURES: ")
        print(np.fromstring(person.face_features, int))

        cv2.destroyAllWindows()
        break

    # Quit the program with q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
