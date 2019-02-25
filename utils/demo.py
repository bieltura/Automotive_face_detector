import cv2
import time


def demo_face_detector(camera, frame):
    landmarks = camera.getLandmarks()

    if landmarks is not None:
        # Hard code for the landmarks
        for i, (x, y) in enumerate(landmarks):

            # Small 5 shape landmakrs
            if i in [36, 39, 45, 42, 33]:

                # Affine transformation
                if i in [36, 45, 33]:
                    cv2.circle(frame, (x, y), 8, (255, 255, 255), -1)
                    cv2.putText(frame, 'Affine transform', (camera.getDim()[0] - 450, camera.getDim()[1] - 200),
                                cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255))

                cv2.circle(frame, (x, y), 5, (255, 0, 255), -1)
                cv2.putText(frame, '5 point shape predictor', (camera.getDim()[0] - 450, camera.getDim()[1] - 150),
                            cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 255))
            else:
                cv2.circle(frame, (x, y), 4, (192, 162, 103), -1)
                cv2.putText(frame, '68 point shape predictor', (camera.getDim()[0] - 450, camera.getDim()[1] - 100),
                            cv2.FONT_HERSHEY_TRIPLEX, 1, (192, 162, 103))

    return frame


def compute_fps(num_frames, start):
    end = time.time()
    fps = num_frames / (end - start)
    return fps, end


def demo_fps(camera, frame, fps):
    cv2.putText(frame, '{0:.1f} FPS'.format(fps), (100, camera.getDim()[1] - 100),
                cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255))

    return frame
