import cv2
from obj import devices
from face_detector import CameraFaceDetector
from utils import image_processing as ip

# Cameras (to be written into a file)
num_cameras = 2

# Face size (square in px for CNN)
face_size = 250

# Number of cameras in the system
cameras = [None] * num_cameras
camera_thread = []

# Still working on a good image
text = cv2.imread('res/text_detect.png')

for cam_id in range(num_cameras):
    cameras[cam_id] = devices.FaceCamera(cam_id)
    thread = CameraFaceDetector(cameras[cam_id], face_size)
    thread.setName('Cam '+str(cam_id))
    camera_thread.append(thread)
    thread.start()

# Main program
while True:

    for cam_id, camera in enumerate(cameras):

        frame = camera.getFrame()
        face = camera.getFace()

        if frame is not None:
            cv2.imshow(str(cam_id), cv2.resize(camera.getFrame(), tuple(int(x/2) for x in camera.getDim())))

        if face is not None:
            print("Face detected in camera " + str(cam_id))

            frames = ip.mask(frame, text)

            cv2.imshow(str(cam_id), cv2.resize(frames, tuple(int(x/2) for x in camera.getDim())))

            # Calls to the NN HERE

            # Restart face detector (same camera)
            camera_thread[cam_id] = CameraFaceDetector(cameras[cam_id], face_size)
            camera_thread[cam_id].start()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        for thread in camera_thread:
            thread.stop()
        cv2.destroyAllWindows()

        break
