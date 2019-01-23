import database
import numpy as np


database.create_database()

for i in range(9):
    database.add_person("Nombre " + str(i), "None")


persons = database.get_all_persons()

for person in persons:
    print(" ")
    print("PERSON ID " + str(person.id))
    print("NAME: " + str(person.name))
    print("PICTURE: " + str(person.face_img_path))
    print("FACE FEATURES:")
    print(np.fromstring(person.face_features,int))
