import numpy as np
import cv2
import os

# Termination criteria
criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Arrays to store object points and image points from all img
objpoints = []
imgpointsR = []
imgpointsL = []

##===========================================================
filenameL= os.path.join("stereo_vision/calibration/", "{}.npy".format("imgpointsL"))
filenameR = os.path.join("stereo_vision/calibration/", "{}.npy".format("imgpointsR"))
filename_op = os.path.join("stereo_vision/calibration/", "{}.npy".format("objpoints"))
filename_mtR = os.path.join("stereo_vision/calibration/", "{}.npy".format("mtxR"))
filename_dR = os.path.join("stereo_vision/calibration/", "{}.npy".format("distR"))
filename_mtL = os.path.join("stereo_vision/calibration/", "{}.npy".format("mtxL"))
filename_dL = os.path.join("stereo_vision/calibration/", "{}.npy".format("distL"))
filename_chR = os.path.join("stereo_vision/calibration/", "{}.npy".format("ChessImaR"))

# Read
imgpointsR = np.load(filenameR)
imgpointsL = np.load(filenameL)
objpoints = np.load(filename_op)
mtxR = np.load(filename_mtR)
distR = np.load(filename_dR)
mtxL = np.load(filename_mtL)
distL = np.load(filename_dL)
ChessImaR = np.load(filename_chR)

calibration_size = (640, 360)

# StereoCalibrate function
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC

retS, MLS, dLS, MRS, dRS, R, T, E, F = cv2.stereoCalibrate(objectPoints=objpoints,
                                                           imagePoints1=imgpointsL,
                                                           imagePoints2=imgpointsR,
                                                           cameraMatrix1=mtxL,
                                                           distCoeffs1=distL,
                                                           cameraMatrix2=mtxR,
                                                           distCoeffs2=distR,
                                                           imageSize=ChessImaR.shape[::-1],
                                                           criteria=criteria_stereo,
                                                           flags=flags)

# if 0 image croped, if 1 image nor croped
rectify_scale = 0
RL, RR, PL, PR, Q, roiL, roiR = cv2.stereoRectify(MLS, dLS, MRS, dRS, ChessImaR.shape[::-1], R, T, rectify_scale, (0, 0))

# initUndistortRectifyMap function
Left_Stereo_Map = cv2.initUndistortRectifyMap(MLS, dLS, RL, PL, ChessImaR.shape[::-1], cv2.CV_16SC2)
Right_Stereo_Map = cv2.initUndistortRectifyMap(MRS, dRS, RR, PR, ChessImaR.shape[::-1], cv2.CV_16SC2)

# Create StereoSGBM and prepare all parameters
window_size = 5
min_disp = 2
num_disp = 114 - min_disp
stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                               numDisparities=num_disp,
                               blockSize=window_size,
                               uniquenessRatio=10,
                               speckleWindowSize=100,
                               speckleRange=32,
                               disp12MaxDiff=5,
                               preFilterCap=5,
                               P1=8 * 1 * window_size ** 2,
                               P2=32 * 1 * window_size ** 2)


# Used for the filtered image
stereoR = cv2.ximgproc.createRightMatcher(stereo)  # Create another stereo for right this time

# WLS FILTER Parameters
lmbda = 80000
sigma = 1.8

wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)


def detect_3d_face(frameRight, frameLeft, map_size, ROI=None):

    if frameRight is not None and frameLeft is not None:

        # Size of the image
        height_frame, width_frame, channels = frameRight.shape

        # Resize to the calibration
        frameR = cv2.resize(frameRight, calibration_size)
        frameL = cv2.resize(frameLeft, calibration_size)

        # Rectify the img on rotation and alignement
        Left_nice = cv2.remap(frameL, Left_Stereo_Map[0], Left_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        Right_nice = cv2.remap(frameR, Right_Stereo_Map[0], Right_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

        # Convert from color(BGR) to gray
        grayR = cv2.cvtColor(Right_nice, cv2.COLOR_BGR2GRAY)
        grayL = cv2.cvtColor(Left_nice, cv2.COLOR_BGR2GRAY)

        # Filter the noise to make the stereo match
        grayR = cv2.fastNlMeansDenoising(grayR, None, h=4)
        grayL = cv2.fastNlMeansDenoising(grayL, None, h=4)

        # Compute para el stereo
        dispL = stereo.compute(grayL, grayR)
        dispR = stereoR.compute(grayR, grayL)

        # If a face ROI is given
        if ROI is not None:

            # Disparity map left, left view, filtered_disparity map, disparity map right
            filteredImg = wls_filter.filter(dispL, grayL, None, dispR)
            filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
            filteredImg = np.uint8(filteredImg)

            # Resize the ROI to the disparity map
            top = int(ROI.top() * calibration_size[1] / height_frame)
            bottom = int(ROI.bottom() * calibration_size[1] / height_frame)
            left = int(ROI.left() * calibration_size[0] / width_frame)
            right = int(ROI.right() * calibration_size[0] / width_frame)

            # Get the face with ROI
            filtered_face = gamma_correction(filteredImg[top:bottom, left:right], c=1, y=7)

            # If the ROI is outside the legal borders, return black depth map
            if filtered_face is not None:
                filtered_face = cv2.resize(filtered_face, (map_size, map_size))
            else:
                filtered_face = np.zeros((map_size,map_size), np.uint8)

            return filtered_face, filteredImg

        return None, None


def gamma_correction(img, c, y):
    if not any(dim is 0 for dim in img.shape):
        img = img + (255 - np.max(img))
        gamma = 255 * c * ((img/255) ** y)
        return gamma

    return None
