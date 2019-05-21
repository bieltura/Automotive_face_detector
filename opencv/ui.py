import cv2
from devices import cameras
from main import Recognizer
import numpy as np

stereo = True

if stereo:
    num_cameras = 2
    wallpaper = cv2.imread("img/ui_3d.png")
else:
    num_cameras = 1
    wallpaper = cv2.imread("img/ui_2d.png")

cam = [None] * num_cameras

# Setup the cameras
for cam_id in range(num_cameras):
    cam[cam_id] = cameras.FaceCamera(cam_id)
    cam[cam_id].start()

recognizer = Recognizer(cam=cam[0], stereo=stereo)

detected_face_landmarks = None
detected_face = None
detected_map = None
match = None

# Method that draws the output
def draw(stereo, results, frame_right, frame_left):
    global detected_face_landmarks, detected_face, detected_map, match
    x_margin = 24
    y_margin = 70

    frame_width = 504
    frame_height = int(frame_width * 9 / 16)

    square_images = 240

    ui = wallpaper.copy()

    if frame_right is not None:
        ui[y_margin:y_margin+frame_height, x_margin:x_margin+frame_width] = cv2.resize(frame_right, (frame_width, frame_height))

    if results[0] is not None:
        detected_face_landmarks = results[0]

    if detected_face_landmarks is not None:
        ui[2*y_margin + frame_height:2*y_margin + frame_height + square_images, x_margin:x_margin+square_images] = cv2.resize(detected_face_landmarks, (square_images,square_images))

    if results[1] is not None:
        detected_face = results[1]

    if detected_face is not None:
        ui[2*y_margin + frame_height:2*y_margin + frame_height + square_images, 2*x_margin+square_images:2*x_margin+2*square_images] = cv2.resize(detected_face, (square_images,square_images))

    if stereo:
        if frame_left is not None:
            ui[y_margin:y_margin + frame_height, 2 * x_margin + frame_width:2 * x_margin + 2 * frame_width] = cv2.resize(frame_left, (frame_width, frame_height))

        if results[3] is not None:
            detected_map = cv2.cvtColor(np.uint8(results[3]), cv2.COLOR_GRAY2BGR)

        if detected_map is not None:
            ui[2 * y_margin + frame_height:2 * y_margin + frame_height + square_images, 3 * x_margin + 2 * square_images:3 * x_margin + 3 * square_images] = cv2.resize(detected_map, (square_images, square_images))

    if results[2] is not None:
        match = results[2]

    if match is not None:
        if match is "unknown":
            match = "I can't recognize you"
        cv2.putText(ui, match, (3*square_images + 3*x_margin + 20, 424 + 30), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0,0,0))

    return ui


# Main method
while True:
    frame_right = cam[0].getFrame()

    if stereo:
        frame_left = cam[1].getFrame()
    else:
        frame_left = None

    # Check if the frame right is captures
    if frame_right is not None:
        results = recognizer.recognize(stereo, frame_right, frame_left)
        ui = draw(stereo, results, frame_right, frame_left)
        cv2.imshow('3D Facial recognition', ui)

    # Closing the app
    if cv2.waitKey(1) & 0xFF == ord('q'):

        # Camera thread stop
        for camera in cam:
            camera.close()

        # Recognizer thread
        recognizer.close()

        cv2.destroyAllWindows()
        break
