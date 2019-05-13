import cv2
import time
import numpy as np


def landmarks_img(frame, landmarks, bb, face_size):

    if frame is not None:
        if landmarks is not None:
            # Hard code for the landmarks
            for i, (x, y) in enumerate(landmarks):

                # Small 5 shape landmakrs
                #if i in [36, 39, 45, 42, 33]:

                    # Affine transformation
                #    if i in [36, 45, 33]:
                        #cv2.circle(frame, (x, y), 8, (255, 255, 255), -1)

                 #   cv2.circle(frame, (x, y), 5, (255, 0, 255), -1)
                #else:
                if y < frame.shape[0] and x < frame.shape[1]:
                    cv2.circle(frame, (x, y), 5, (192, 162, 103), -1)

        if bb is not None:
            frame_face = frame[bb.top():bb.bottom(), bb.left():bb.right()].copy()

            # The borders
            if not any(dim is 0 for dim in frame_face.shape):
                frame_face = cv2.resize(np.uint8(frame_face), (face_size, face_size))

            else:
                frame_face = np.zeros((face_size,face_size,3), np.uint8)

    return frame_face


def compute_fps(num_frames, start):
    end = time.time()
    fps = num_frames / (end - start)
    return fps, end


def demo_fps(camera, frame, fps):
    cv2.putText(frame, '{0:.1f} FPS'.format(fps), (100, camera.getDim()[1] - 100),
                cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255))

    return frame
