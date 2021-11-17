import cv2
import numpy as np
import imutils
import math
import mahotas
import matplotlib.pyplot as plt

def getContours(image, lowThresholdDist=0.1):
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
    imgLaplacian = cv2.filter2D(image, cv2.CV_32F, kernel)
    sharp = np.float32(image)
    imgResult = sharp - imgLaplacian
    imgResult = np.clip(imgResult, 0, 255)
    imgResult = imgResult.astype('uint8')

    dist = cv2.distanceTransform(image, cv2.DIST_L2, 3,cv2.DIST_MASK_PRECISE )
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    hist = cv2.calcHist(dist, [0], None, [255], [0,1])
    _, dist = cv2.threshold(dist, lowThresholdDist, 1.0, cv2.THRESH_BINARY)
    kernel1 = np.ones((3,3), dtype=np.uint8)
    dist = cv2.dilate(dist, kernel1)
    dist_8u = dist.astype('uint8')
    # Find total markers
    contours = cv2.findContours(dist_8u, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    return contours

def extractGeometricalProperties(contour, mask):
    shapeDict = {}
    # Moments
    M = cv2.moments(contour)
    shapeDict['xTile'] = int(M["m10"] / M["m00"])
    shapeDict['yTile'] = int(M["m01"] / M["m00"])
    # perimeter properties : perimeter of the contour; perimeter of a shape approximation
    perimeter = cv2.arcLength(contour,True)
    shapeDict['perimeter'] = perimeter
    epsilon = 0.10 * perimeter
    perimeterApproximation = cv2.arcLength(cv2.approxPolyDP(contour,epsilon,True),True)
    shapeDict['perimeterApproximation'] = perimeterApproximation
    complexity = perimeter/perimeterApproximation
    shapeDict['complexity'] = complexity
    # area properties : area of the contour: area of the minimal enclosing circle
    area = cv2.contourArea(contour)
    shapeDict['area'] = area
    _, radiusEnclosingCircle = cv2.minEnclosingCircle(contour)
    areaCircle = math.pi * radiusEnclosingCircle * radiusEnclosingCircle
    circleness = area/areaCircle
    shapeDict['circleness'] = circleness
    (x, y, w, h) = cv2.boundingRect(contour)
    shapeDict['W'] = w
    shapeDict['H'] = h
    shapeDict['rectangleness'] = area / (w*h)
    solidity = (area / cv2.contourArea(cv2.convexHull(contour, returnPoints=True)))
    shapeDict['solidity'] = solidity
    # Decomposition
    ROI = mask[y:y + h, x:x + w]
    zernikeDecomposition = mahotas.features.zernike_moments(ROI, radiusEnclosingCircle, degree=8)
    for index, value in enumerate(zernikeDecomposition):
        shapeDict[f'zernikeDecomposition_{index}'] = value
    return shapeDict