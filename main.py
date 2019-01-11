import cv2

from face_detector import CameraFaceDetector
from obj import devices

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

            detection_text = cv2.resize(text, camera.getDim())

            detection_text_gray = cv2.cvtColor(detection_text, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(detection_text_gray, 20, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            # Now black-out the area of logo in ROI
            img1_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)

            # Take only region of logo from logo image.
            img2_fg = cv2.bitwise_and(detection_text, detection_text, mask=mask)

            # Put text in ROI and modify the main image
            frame = cv2.add(img1_bg, img2_fg)

            cv2.imshow(str(cam_id), cv2.resize(frame, tuple(int(x/2) for x in camera.getDim())))

            # Calls to the NN HERE

            # Restart camera service
            cameras[cam_id] = devices.FaceCamera(cam_id)
            thread = CameraFaceDetector(cameras[cam_id], face_size)
            thread.setName('Cam ' + str(cam_id))
            camera_thread[cam_id] = thread
            thread.start()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        for thread in camera_thread:
            thread.stop()
        cv2.destroyAllWindows()

        break
