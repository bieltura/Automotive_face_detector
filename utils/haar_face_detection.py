import cv2
from threading import Lock
lock = Lock()

face_cascade = cv2.CascadeClassifier("data/haar/haarcascade_frontalface_default.xml")

face_scale_factor = 1/5
minNeighbour = 5
face_pyramid_factor = 1.3


def detect_face(frame, width_face, height_face):
	# Size of the image
	width_frame, height_frame, channels = frame.shape

	# Minimum size to detect face (50x50 px of face), 250 px from dmax.
	frame_face = cv2.resize(frame, (int(height_frame * face_scale_factor), int(width_frame * face_scale_factor)))

	face = get_face(frame_face, width_face, height_face)

	return face


# Returns the face crop of an image in the specified size
def get_face(frame, width, height):

	# Convert the frame to YUV ColorSpace
	frame_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)[:,:,0]

	lock.acquire()
	try:
		# Detection of the face - return rectangle (x,y),(x+w,y+h) (can be done once by the thread)
		faces = face_cascade.detectMultiScale(frame_yuv, face_pyramid_factor, minNeighbour)
	finally:
		lock.release()  # release lock

	for (x, y, w, h) in faces:

		# Cut the face part
		face = frame[y:y+h, x:x+w, :]

		# Resize to fit the neural network
		face = cv2.resize(face, (height, width), interpolation=cv2.INTER_CUBIC)
	
		return face

	return None
