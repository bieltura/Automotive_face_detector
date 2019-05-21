import cv2
import dlib
import numpy as np

# Template for 68 different points in the face to perform the affine transformation
BIG_TEMPLATE = np.float32([
    (0.079239691381, 0.339223741112), (0.082921948723, 0.456955367943),
    (0.096792710916, 0.575648016728), (0.122141515615, 0.691921601066),
    (0.168687863544, 0.800341263616), (0.239789390707, 0.895732504778),
    (0.325662452515, 0.977068762493), (0.422318282013, 1.043290001490),
    (0.531777802068, 1.060803711260), (0.641296298053, 1.039819241070),
    (0.738105872266, 0.972268833998), (0.824444363295, 0.889624082279),
    (0.894792677532, 0.792494155836), (0.939395486253, 0.681546643421),
    (0.961119338290, 0.562238253072), (0.970579841181, 0.441758925744),
    (0.971193274221, 0.322118743967), (0.163846223133, 0.249151738053),
    (0.217803546570, 0.204255863861), (0.291299351124, 0.192367318323),
    (0.367460241458, 0.203582210627), (0.439294511300, 0.233135599851),
    (0.586445962425, 0.228141644834), (0.660152671635, 0.195923841854),
    (0.737466449096, 0.182360984545), (0.813236546239, 0.192828009114),
    (0.870757188600, 0.235293377042), (0.515345338270, 0.318635461930),
    (0.516221448289, 0.396200446263), (0.517118861835, 0.473797687758),
    (0.518164303430, 0.553157797772), (0.433701156035, 0.604054457668),
    (0.475501237769, 0.620763440240), (0.520712933176, 0.634268222208),
    (0.565874114041, 0.618796581487), (0.607054002672, 0.601576716560),
    (0.252418718401, 0.331052263829), (0.298663015648, 0.302646354002),
    (0.355749724218, 0.303020650651), (0.403718978315, 0.338677110830),
    (0.352507175597, 0.349987615384), (0.296791759886, 0.350478978225),
    (0.631326076346, 0.334136672344), (0.679073381078, 0.296454042670),
    (0.735972361530, 0.294721285802), (0.782865376271, 0.321305281656),
    (0.740312274764, 0.341849376713), (0.684998500910, 0.343734332172),
    (0.353167761422, 0.746189164237), (0.414587777921, 0.719053835073),
    (0.477677654595, 0.706835892494), (0.522732900812, 0.717092275768),
    (0.569832064287, 0.705414478982), (0.635195811927, 0.715655725160),
    (0.699516723310, 0.739419187253), (0.639447159575, 0.805236879972),
    (0.576410514055, 0.835436670169), (0.525398405766, 0.841706377792),
    (0.476415457690, 0.837505914975), (0.413795489020, 0.810045601727),
    (0.380084785646, 0.749979603086), (0.477955996282, 0.745132346120),
    (0.523389793327, 0.748924302636), (0.571057789237, 0.743328946910),
    (0.672409137852, 0.744177032192), (0.572539621444, 0.776609286626),
    (0.524010650300, 0.783370783245), (0.477561227414, 0.778476346951)])

# Template for 5 points to perform the affine transformation
SMALL_TEMPLATE = BIG_TEMPLATE[[45, 42, 36, 39, 33]]

#predictor = dlib.shape_predictor('face_detector/data/dlib/shape_predictor_5_face_landmarks.dat')
#landmarkIndices = [0, 2, 4]
#TEMPLATE = SMALL_TEMPLATE

predictor = dlib.shape_predictor('face_detector/data/dlib/shape_predictor_68_face_landmarks.dat')
landmarkIndices = [36, 45, 33]
TEMPLATE = BIG_TEMPLATE

# Normalization process
tpl_min, tpl_max = np.min(BIG_TEMPLATE, axis=0), np.max(BIG_TEMPLATE, axis=0)
TEMPLATE = (TEMPLATE - tpl_min) / (tpl_max - tpl_min)
del BIG_TEMPLATE


# Transform and align a face in an image
def align(imgDim, rgbImg, bb=None):

    # Get the face Bounding box of the frame
    if bb is None:
        return None

    # Get the landmarks of the face
    landmarks = findLandmarks(rgbImg, bb)
    npLandmarks = np.float32(landmarks)
    npLandmarkIndices = np.array(landmarkIndices)

    # Create a 2x3 matrix to transform lines (3 points) - TEMPLATE is 1px x 1px
    H = cv2.getAffineTransform(npLandmarks[npLandmarkIndices], imgDim * TEMPLATE[npLandmarkIndices])

    # Apply the affine transformation
    thumbnail = cv2.warpAffine(rgbImg, H, (imgDim, imgDim))

    # Return the landmarks
    return thumbnail, npLandmarks, bb


# Find the landmarks of a face
def findLandmarks(rgbImg, bb):
    points = predictor(rgbImg, bb)
    return list(map(lambda p: (p.x, p.y), points.parts()))
