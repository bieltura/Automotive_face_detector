import numpy as np
import cv2
import os

# Termination criteria
criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Arrays to store object points and image points from all images
objpoints = []  # 3d points in real world space
imgpointsR = []  # 2d points in image plane
imgpointsL = []

##===========================================================
filenameL = os.path.join("stereo/models/", "{}.npy".format("imgpointsL"))
filenameR = os.path.join("stereo/models/", "{}.npy".format("imgpointsR"))
filename_op = os.path.join("stereo/models/", "{}.npy".format("objpoints"))
filename_mtR = os.path.join("stereo/models/", "{}.npy".format("mtxR"))
filename_dR = os.path.join("stereo/models/", "{}.npy".format("distR"))
filename_mtL = os.path.join("stereo/models/", "{}.npy".format("mtxL"))
filename_dL = os.path.join("stereo/models/", "{}.npy".format("distL"))
filename_chR = os.path.join("stereo/models/", "{}.npy".format("ChessImaR"))

# Read
print(filenameR)
imgpointsR = np.load(filenameR)
imgpointsL = np.load(filenameL)
objpoints = np.load(filename_op)
mtxR = np.load(filename_mtR)
distR = np.load(filename_dR)
mtxL = np.load(filename_mtL)
distL = np.load(filename_dL)
ChessImaR = np.load(filename_chR)

print('Cameras Ready to use')

# ********************************************
# ***** Calibrate the Cameras for Stereo *****
# ********************************************

# StereoCalibrate function
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC

retS, MLS, dLS, MRS, dRS, R, T, E, F = cv2.stereoCalibrate(objpoints,
                                                           imgpointsL,
                                                           imgpointsR,
                                                           mtxL,
                                                           distL,
                                                           mtxR,
                                                           distR,
                                                           ChessImaR.shape[::-1],
                                                           criteria_stereo,
                                                           flags)

# StereoRectify function
rectify_scale = 0  # if 0 image croped, if 1 image nor croped
RL, RR, PL, PR, Q, roiL, roiR = cv2.stereoRectify(MLS, dLS, MRS, dRS,
                                                  ChessImaR.shape[::-1], R, T,
                                                  rectify_scale,
                                                  (0, 0))  # last paramater is alpha, if 0= croped, if 1= not croped
# initUndistortRectifyMap function
Left_Stereo_Map = cv2.initUndistortRectifyMap(MLS, dLS, RL, PL,
                                              ChessImaR.shape[::-1],
                                              cv2.CV_16SC2)  # cv2.CV_16SC2 this format enables us the programme to work faster
Right_Stereo_Map = cv2.initUndistortRectifyMap(MRS, dRS, RR, PR,
                                               ChessImaR.shape[::-1], cv2.CV_16SC2)

# *******************************************
# ***** Parameters for the StereoVision *****
# *******************************************

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
lmbda = 80000  # 80000
sigma = 1.8  # 1.8

wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

# *************************************
# ***** Starting the StereoVision *****
# *************************************

# Call the two cameras
#CamR = cv2.VideoCapture(0)  # Wenn 0 then Right Cam and wenn 2 Left Cam
#CamL = cv2.VideoCapture(1)

def detect_3d_face(frameR, frameL, ROI=None):

    frameR = cv2.resize(frameR, (480, 320))
    frameL = cv2.resize(frameL, (480, 320))

    # Rectify the images on rotation and alignement
    # Rectify the image using the calibration parameters founds during the initialisation
    Left_nice = cv2.remap(frameL, Left_Stereo_Map[0], Left_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    Right_nice = cv2.remap(frameR, Right_Stereo_Map[0], Right_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

    # Convert from color(BGR) to gray
    grayR = cv2.cvtColor(Right_nice, cv2.COLOR_BGR2GRAY)
    grayL = cv2.cvtColor(Left_nice, cv2.COLOR_BGR2GRAY)

    # Compute the 2 images for the Depth_image

    # Compute para el stereo
    dispL = stereo.compute(grayL, grayR)
    dispR = stereoR.compute(grayR, grayL)

    # Using the WLS filter

    # Disparity map left, left view, filtered_disparity map, disparity map right, ROI=rect (to be done)
    filteredImg = wls_filter.filter(dispL, grayL, None, dispR, ROI=ROI)
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)

    # Change the Color of the Picture into an Ocean Color_Map
    # filt_Color = cv2.applyColorMap(filteredImg,cv2.COLORMAP_OCEAN)

    return True
