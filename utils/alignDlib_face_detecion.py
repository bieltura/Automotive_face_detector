import cv2
from threading import Thread
import dlib

from utils.align import AlignDlib

# Scale factor! Adaptive to the dimensions!!!
face_scale_factor = 1/15


class Detector(Thread):
	def __init__(self):
		Thread.__init__(self)

		self.frame = None
		self.face = None

		# Cascade needs to be loaded for every different camera
		self.alignment = AlignDlib('data/dlib/shape_predictor_68_face_landmarks.dat')

		# Variable to stop the camera thread if needed
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
					#frame_yuv = cv2.cvtColor(frame_face, cv2.COLOR_BGR2YUV)[:, :, 0]

					# Detect the bounding box
					bb = self.alignment.getLargestFaceBoundingBox(frame_face)

					if bb is None:
						self.face = None

					else:

						# Re-scale the values to cut from original frame
						x, y, w, h = (int(var / face_scale_factor) for var in (bb.left(), bb.top(), bb.width(), bb.height()))

						# Create the new boundary box
						bb = dlib.rectangle(left=x, top=y, right=x+w, bottom=y+h)

						# Crop the image from the boundary box (the big)
						face_aligned = self.alignment.align(250, self.frame, bb, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
						self.face = face_aligned

					# Deactivate the loop, waiting for new frame
					self.frame = None
			else:
				# Break the loop and stop the thread
				return

	def detect_face(self, frame):
		# Active the loop to analyze the frame
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
