# ****************************************************************************
# *  Script to calibrate thermal camera to obtain intrinsic and distortion matrices.
# *  Uses a symmetric circle calibration sheet obtained on https://calib.io/pages/camera-calibration-pattern-generator

import cv2
import numpy as np
import os
import glob

CIRCLE_PATTERN = (8,8)
subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
objp = np.zeros((1, CIRCLE_PATTERN[0]*CIRCLE_PATTERN[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CIRCLE_PATTERN[0], 0:CIRCLE_PATTERN[1]].T.reshape(-1, 2)
_img_shape = None
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('data/circles/*.jpg') # file path
gray = []
for fname in images:
    img = cv2.imread(fname)
    if _img_shape == None:
        _img_shape = img.shape[:2]
    else:
        assert _img_shape == img.shape[:2], "All images must share the same size."
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)

    # cv2.imshow('gray', gray)
    # cv2.waitKey(0)
 
    # Find the circles
    ret, circles= cv2.findCirclesGrid(gray, CIRCLE_PATTERN, None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        # corners2 = cv2.cornerSubPix(gray,circles,(3,3),(-1,-1),subpix_criteria)
        imgpoints.append(circles)
        draw = cv2.drawChessboardCorners(img, CIRCLE_PATTERN, circles, ret)
        cv2.imshow('test', draw)
        # cv2.imshow('gray', gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print(fname)
N_OK = len(objpoints)
K = np.zeros((3, 3))
D = np.zeros((1,4))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

ret, mtx, dist_coeff, R_vecs, T_vecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("Found " + str(N_OK) + " valid images for calibration")
print("DIM=" + str(_img_shape[::-1]))
print("K=np.array(" + str(mtx.tolist()) + ")")
print("D=np.array(" + str(dist_coeff.tolist()) + ")")
