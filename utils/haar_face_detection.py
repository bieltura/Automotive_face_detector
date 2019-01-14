import cv2
from threading import Thread

# Haar filter values
face_scale_factor = 1/6
minNeighbour = 5
face_pyramid_factor = 1.3


class Detector(Thread):
	def __init__(self):
		Thread.__init__(self)

		self.frame = None
		self.face = None

		# Cascade needs to be loaded for every different camera
		self.face_cascade = cv2.CascadeClassifier("data/haar/haarcascade_frontalface_default.xml")

		self.stopThread = False

	def run(self):
		while True:
			if not self.stopThread:
				if self.frame is not None:

					# Size of the image
					height_frame, width_frame, channels = self.frame.shape

					# Minimum size to detect face (50x50 px of face), 250 px from dmax.
					frame_face = cv2.resize(self.frame, (int(width_frame * face_scale_factor), int(height_frame * face_scale_factor)))

					# Convert the frame to YUV ColorSpace
					frame_yuv = cv2.cvtColor(frame_face, cv2.COLOR_BGR2YUV)[:, :, 0]

					# Detection of the face - return rectangle [(x,y), (x+w,y+h)]
					faces = self.face_cascade.detectMultiScale(frame_yuv, face_pyramid_factor, minNeighbour)

					# No face detected
					if not len(faces):
						self.face = None

					else:
						for (x, y, w, h) in faces:
							# Cut the face part
							self.face = self.frame[y:y + h, x:x + w, :]

					self.frame = None
			else:
				# Break the loop and stop the thread
				return

	def detect_face(self, frame):
		self.frame = frame

	# Returns the face crop of an image in the specified size
	def get_face(self, height_face, width_face):

		if self.face is not None:
			# Resize to fit the neural network
			return cv2.resize(self.face, (height_face, width_face), interpolation=cv2.INTER_CUBIC)
		else:
			return None

	# State variable for stopping face detector service
	def stop(self):
		self.stopThread = True
