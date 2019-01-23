from database import db_service as db
import numpy as np
import cv2
import os


db.create_database()

db.add_person("Biel Tura")


persons = db.get_all_persons()


for person in persons:
    print(" ")
    print("PERSON ID " + str(person.id))
    print("NAME: " + str(person.name))
    print("PICTURE: " + str(person.face_img_path))
    print("FACE FEATURES:")
    print(np.fromstring(person.face_features,int))

img_path = "./face_detector.py"
#img = cv2.imread(img_path,0)
text = cv2.imread('res/text_detect.png')
cv2.imshow("hola",text)



#for person in database.get_persons_by_name("Nombre 3"):
#    print(person.id)
