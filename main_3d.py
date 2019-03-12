import cv2
import glob
from obj import devices
from face_detector import CameraFaceDetector
from face_recognition import FacialRecognition

# Demonstration values:
face_detector_demo = False
face_recognition_demo = False

# Check number of cameras in Linux distribution cameras
num_cameras = 0
for camera in glob.glob("/dev/video?"):
    num_cameras += 1

# Face size (square in px for CNN)
face_size = 96

# Number of cameras in the system
cameras = [None] * num_cameras
face_detected = False

# Currently one face_features for n cameras
face_features = None

# Neural Net Facial recognition start
#facial_recognition_thread = FacialRecognition()
#facial_recognition_thread.start()

# Setup the cameras
for cam_id in range(num_cameras):
    cameras[cam_id] = devices.FaceCamera(cam_id)
    cameras[cam_id].start()

face_detector = CameraFaceDetector(cameras[cam_id].getScaleFactor(), face_size, stereo=True)
face_detector.start()

# Main program
while True:

    # Get the frame from the camera
    frame = cameras[0].getFrame()
    frame_sec = cameras[1].getFrame()

    # If we have not detected any face
    if not face_detected:

        if frame is not None:
            face_detector.detect(frame, frame_sec)

            face = face_detector.getFace()
            face_3d = face_detector.get3dFace()

            if face is not None:
                print("Face detected in camera " + str(cam_id))

                # Pause the face detector thread by setting a Nonr frame
                face_detector.detect(None)

                # face_detected = True
                if face_3d is not None:
                    cv2.imshow("Face " + str(cam_id), face_3d)

    if frame is not None:
        cv2.imshow("Camera " + str(cam_id), cv2.resize(frame, tuple(int(x * cameras[0].getScaleFactor() * 5) for x in cameras[0].getDim())))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Face detector thread stop
        face_detector.stop()

        # Camera thread stop
        for camera in cameras:
            camera.close()

        cv2.destroyAllWindows()
        break
