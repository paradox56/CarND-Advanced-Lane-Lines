import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.
d = 0
# Make a list of calibration images
images = glob.glob('../camera_cal/calibration*.jpg')

def camera_calibration_matrix(objpoints, imgpoints, img):
    #This function takes chessboard images with known dimensions and obtain camera calibration matrix
    localImg = np.copy(img)
    grayImg = cv2.cvtColor(localImg,cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tves = cv2.calibrateCamera(objpoints, imgpoints, grayImg.shape[::-1],None,None)
    return mtx,dist
