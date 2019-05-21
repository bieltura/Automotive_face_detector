import cv2
import PIL
from PIL import ImageTk, Image
from tkinter import *
from face_detector import dlib_face_detecion as detector
from devices import cameras
from face_recognition.recognition import FacialRecognition
from database import db_service as db
from tkinter import messagebox


def run(old_root):
    frame_width = 504
    frame_height = int(frame_width * 9 / 16)
    frame = None

    # Open the camera to capture the picture
    camera = cameras.FaceCamera(0)
    camera.start()

    # Open the database
    db.create_database()

    # Neural Net Facial recognition start
    facial_recognition_thread = FacialRecognition(stereo=False)
    facial_recognition_thread.start()

    # User Interface
    root = Toplevel(old_root)
    root.configure(background='ghost white')

    # Camera image
    cam_label = Label(root)
    cam_label.grid(row=0, columnspan=3, pady=(20, 20), padx=(20, 20))

    # Name text
    Label(root, text="Name: ", background='ghost white').grid(row=1, column=0, pady=(0,20), padx=(20, 0))

    # Name text entry
    name_entry = Entry(root, width=40, background='ghost white')
    name_entry.grid(row=1, column=1, pady=(0, 20))

    def register():
        global frame

        name = name_entry.get()

        if frame is not None and name is not "":

            # Get the aligned face
            alignedFace, _, _ = detector.detect_face(frame, 240, camera.getScaleFactor())

            picture_path = "database/img/" + name.replace(" ", "_") + ".jpg"

            # Save the image of the registered face
            cv2.imwrite(picture_path, alignedFace)

            # Recognize the face
            facial_recognition_thread.recognize_face(alignedFace)

            face_features = None

            # Wait for the face features to be computed
            while face_features is None:
                face_features = facial_recognition_thread.get_face_features()

            # Create the person in the database
            db.add_person(name, picture_path, face_features)

            # Print the person registered:
            person = db.get_persons_by_name(name)

            messagebox.showinfo("Registration", person.name + " has been registered.")

    # Register button
    register_button = Button(root, text="Register", command=register, width=10, height=2)
    register_button.grid(row=1, column=2, pady=(0, 20), padx=(20, 20))

    def show_frame():
        global frame
        frame = camera.getFrame()

        if frame is not None:

            img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA), (frame_width,frame_height))
            img = PIL.Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)

            cam_label.imgtk = imgtk
            cam_label.configure(image=imgtk)

        cam_label.after(30, show_frame)

    def close():
        camera.close()
        facial_recognition_thread.stop()
        root.destroy()

    show_frame()

    root.protocol("WM_DELETE_WINDOW", close)
    root.mainloop()
