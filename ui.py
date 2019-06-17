import cv2
from devices import cameras
from main import Recognizer
import numpy as np

detected_face_landmarks = None
detected_face = None
detected_map = None
match = None

wallpaper = None


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

        if stereo:
            x = 3*square_images + 3*x_margin + 100
            y = 3*y_margin + frame_height + 50
            size = 60
        else:
            x = int(frame_width / 2)
            y= frame_height + 2*y_margin + square_images + 20
            size = 30

        if match is "unknown" or match is None:
            action = cv2.resize(cv2.imread("img/lock.jpg"), (size, size))
            match_text = "Identity not recognized"
        else:
            match_text = match
            action = cv2.resize(cv2.imread("img/unlock.jpg"), (size, size))

        ui[y:y+size, x:x+size] = action

        if stereo:
            cv2.putText(ui, match_text, (x - 60, y - 30), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 0, 0))
        else:
            cv2.putText(ui, match_text, (x + 2*size, y+20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 0))

    return ui


def run(stereo):
    global wallpaper
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

    recognizer = Recognizer(scaleFactor=cam[0].getScaleFactor(), stereo=stereo)


    # Main method
    while True:
        frame_right = cam[0].getFrame()

        if stereo:
            title  = "3D Face recognition"
            frame_left = cam[1].getFrame()
        else:
            title = "2D Face recognition"
            frame_left = None

        # Check if the frame right is captures
        if frame_right is not None:
            results = recognizer.recognize(stereo, frame_right, frame_left)
            ui = draw(stereo, results, frame_right, frame_left)
            cv2.imshow(title, ui)

        # Closing the app
        if cv2.waitKey(1) & 0xFF == ord('q'):

            # Camera thread stop
            for camera in cam:
                camera.close()

            # Recognizer thread
            recognizer.close(stereo)

            cv2.destroyAllWindows()
            break
