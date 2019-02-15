import cv2
import dlib

from utils.dlib_face_alignment import AlignDlib

# Scale factor! Adaptive to the dimensions!!!
face_scale_factor = 1/15

# Cascade needs to be loaded for every different camera
alignment = AlignDlib('data/dlib/shape_predictor_68_face_landmarks.dat')


def detect_face(frame, face_dim):
	if frame is not None:

		# Size of the image
		height_frame, width_frame, channels = frame.shape

		# Calculus of face_scale factor
		# face_scale_factor = TODO

		# Minimum size to detect face (50x50 px of face), 250 px from dmax.
		frame_face = cv2.resize(frame, (int(width_frame * face_scale_factor), int(height_frame * face_scale_factor)))

		# Convert the frame to YUV ColorSpace
		frame_yuv = cv2.cvtColor(frame_face, cv2.COLOR_BGR2YUV)[:, :, 0]

		# Detect the bounding box
		bb = alignment.getLargestFaceBoundingBox(frame_yuv)

		if bb is None:
			return None

		else:
			# Re-scale the values to cut from original frame
			x, y, w, h = (int(var / face_scale_factor) for var in (bb.left(), bb.top(), bb.width(), bb.height()))

			# Create the new boundary box
			bb = dlib.rectangle(left=x, top=y, right=x+w, bottom=y+h)

			# Crop the image from the boundary box (the big)
			return alignment.align(face_dim, frame, bb, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

	else:
		# Break the loop and stop the thread
		return None
