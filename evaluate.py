import cv2
from main import Recognizer
import numpy as np

stereo = True

if stereo:
    num_cameras = 2
else:
    num_cameras = 1

recognizer = Recognizer(scaleFactor=1, stereo=stereo)

num_files = 24

# Main method
for file in range(0, num_files):

    frame_right = cv2.imread('evaluation/img/'+str(file)+'_r.png')

    if stereo:
        frame_left = cv2.imread('evaluation/img/'+str(file)+'_l.png')
    else:
        frame_left = None

    results = [None] * 4

    # Check if the frame right is captured
    if frame_right is not None:
        while results[2] is None:
            results = recognizer.recognize(stereo, frame_right, frame_left)

    print("Image {0}: ".format(file))

    # Evaluate the whole matching - MATCH
    print("MATCH: {0}".format(results[2]))


    while True:
        # Evaluate the detected face: - ALIGNED
        #cv2.imshow('Face Detector', results[1])

        # Evaluaate the stereo system:
        if stereo:
            cv2.imshow('Stereo', np.uint8(results[3]))

        k = cv2.waitKey(1)

        # ESC pressed
        if k % 256 == 27:
            print("Escape hit, closing...")
            break


# Recognizer thread
recognizer.close(stereo)
