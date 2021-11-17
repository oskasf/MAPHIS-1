
from __future__ import print_function
import time
'''import cv2
import math
import time
import pandas as pd
import csv
import imutils
import numpy as np

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

t0 = time.time()'''

#tileName = "testShapeDetection.jpg"
#tileName = "1905021030252.png"
"""
t0 = time.time()
image = cv2.imread(tileName, cv2.IMREAD_GRAYSCALE)

kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
# do the laplacian filtering as it is
# well, we need to convert everything in something more deeper then CV_8U
# because the kernel has some negative values,
# and we can expect in general to have a Laplacian image with negative values
# BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
# so the possible negative number will be truncated
imgLaplacian = cv2.filter2D(image, cv2.CV_32F, kernel)
sharp = np.float32(image)
imgResult = sharp - imgLaplacian
# convert back to 8bits gray scale
imgResult = np.clip(imgResult, 0, 255)
imgResult = imgResult.astype('uint8')

dist = cv2.distanceTransform(image, cv2.DIST_L2, 3,cv2.DIST_MASK_PRECISE )
cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)

_, dist = cv2.threshold(dist, 0.1, 1.0, cv2.THRESH_BINARY)

# Dilate a bit the dist image
kernel1 = np.ones((3,3), dtype=np.uint8)
dist = cv2.dilate(dist, kernel1)

dist_8u = dist.astype('uint8')
# Find total markers
contours = cv2.findContours(dist_8u, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
markers = np.zeros(dist.shape, dtype=np.int32)"""

# Draw the foreground markers
'''maxLen = max([len(l) for l in contours])

hist = np.zeros(maxLen)

for cnt in contours:
    hist[len(cnt)-1] +=1

nC = 10
model = GaussianMixture(n_components=nC)
hist = hist.reshape(-1,1)
model.fit(hist)
yhat = model.predict(hist)

t1 = time.time()
def getColor(cat,nC):
    return int(cat*255/nC)

for i, cnt in enumerate(contours):
    cntLen = len(cnt)-1
    cv2.drawContours(markers, contours, i, (getColor(yhat[cntLen],nC)), -1)
# Draw the background marker
cv2.circle(markers, (5,5), 3, (255,255,255), -1)
t2 = time.time()

print(f'Computation time  : {t1-t0}')
print(f'Colorisation time  : {t2-t1}')
print(len(contours))
'''

"""shapeNames = ['field', 'road', 'house', 'hotel', 'forest', 'factory']

featureVars = {'stones' : ['stones'],
                'chimneys' : ['chimneys'],
                'hotels': ['hotels']
              }

featureDictsPath = 'datasets/Barrow_in_Furness/500/tp_2/detected_'

shapeFeatureNames = ['perimeter', 'perimeterApproximation', 'complexity', 'area', 'circleness', 'solidity']
featureNames = ['stones', 'chimneys', 'hotels']
header = ['x', 'y']
for shapeFeatureName in shapeFeatureNames:
    header.append(shapeFeatureName)
for featureName in featureNames:
    header.append(featureName)
print(header)

def extractFeatures(posX, posY, featureDictsPath, tileName):
    featureDict = {}
    for featureName in featureNames:
        featureAggregate = []
        for ftDiff in featureVars[featureName]:
            featureAggregate.append(pd.read_csv(featureDictsPath+ftDiff+'-Barrow_in_Furness.csv'))
        featureAggregateAsDF = pd.concat(featureAggregate)
        queriedDF = featureAggregateAsDF.query('tile == @tileName')
        toleranceRange = (posX**2 + posY**2)**0.5+0.5
        queriedDF = queriedDF.query('(x**2 + y**2)**0.5 <@toleranceRange')
        featureDict[featureName] = len(queriedDF.index)
    return featureDict

def extractGeometricalProperties(contour):
    shapeDict = {}
    # Moments
    M = cv2.moments(contour)
    shapeDict['posX'] = int(M["m10"] / M["m00"])
    shapeDict['posY'] = int(M["m01"] / M["m00"])
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
    solidity = (area / cv2.contourArea(cv2.convexHull(contour, returnPoints=True)))
    shapeDict['solidity'] = solidity
    return shapeDict

def writeDetectionResultsDict(shapeList,  header = header):
    fName = 'test.csv'
    with open(fName,'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(header)

        for detectedElement in shapeList:
            writer.writerow(detectedElement)

w_b = -39428.9058244737
n_b = 22210.1967961792
x_diff = 0.0423510526315792
y_diff = -0.0424076416337286

shapesList = []
for i, contour in enumerate(contours):
    shapeList = []
    shapeDict = extractGeometricalProperties(contour)
    for value in shapeDict.values():
        shapeList.append(value)
    x = w_b+shapeDict['posX']*x_diff
    y = n_b+shapeDict['posY']*y_diff
    localFeatures = extractFeatures(x, y, featureDictsPath, tileName)
    for value in localFeatures.values():
        shapeList.append(value )
    print(shapeList)
    shapesList.append(shapeList)

writeDetectionResultsDict(shapesList)"""
'''
import matplotlib.pyplot as plt
import seaborn as sns
dataframe = pd.read_csv('test.csv', usecols = [i for i in range(2,8)])
dataframe.hist( color='steelblue', edgecolor='black', linewidth=1.0, xlabelsize=8, ylabelsize=8, grid=False)
plt.show()
f, ax = plt.subplots(figsize=(10, 6))
corr = dataframe.corr()
hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
                 linewidths=.05)
f.subplots_adjust(top=0.93)
t= f.suptitle('Shape Attributes Correlation Heatmap', fontsize=14)
plt.show()'''
'''
dataframe = pd.read_csv('results/rawShapes/Barrow_in_Furness/shapeFeatures.csv', usecols = [i for i in range(4,15)])
dataframe.hist( color='steelblue', edgecolor='black', linewidth=1.0, xlabelsize=8, ylabelsize=8, grid=False)
plt.show()
f, ax = plt.subplots(figsize=(10, 6))
corr = dataframe.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
hm = sns.heatmap(round(corr,2), mask=mask,annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
                linewidths=.05)
f.subplots_adjust(top=0.93)
t= f.suptitle('Shape Attributes Correlation Heatmap', fontsize=14)
plt.show()
'''
'''header = ['x', 'y', 'cityName', 'tileName', 'H', 'W', 'xTile', 'yTile', 'perimeter', 'perimeterApproximation',
        'complexity', 'area', 'circleness', 'rectangleness', 'solidity']
for i in range(25):
    header.append(f'zernikeDecomposition_{i}')
'''
'''labelTrain = pd.DataFrame(np.random.randint(2, size=1317))
labelTest = pd.DataFrame(np.random.randint(2, size=167))

dataframeTrain = pd.read_csv('results/rawShapes/Barrow_in_Furness/shapeFeatures.csv', usecols = [i for i in range(4,15)], skiprows=[i for i in range(1300,1450)])
dataframeTest = pd.read_csv('results/rawShapes/Barrow_in_Furness/shapeFeaturesTest.csv', usecols = [i for i in range(4,15)])

dtest = xgb.DMatrix(dataframeTest)
dtrain = xgb.DMatrix(dataframeTrain)
evallist = [(dtest, 'eval'), (dtrain, 'train')]

param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic'}
param['nthread'] = 4
param['eval_metric'] = 'auc'
num_round = 20
bst = xgb.train(param, dtrain, num_round, evallist)

bst.save_model('0001.model')
xgb.to_graphviz(bst, num_trees=2)'''
'''dataframeTrain = pd.read_csv('results/rawShapes/Barrow_in_Furness/shapeFeatures.csv', usecols = [i for i in range(8,15)])
from sklearn.decomposition import PCA
pca = PCA(n_components=6)
pca.fit(dataframeTrain)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)'''

import cv2 as cv
import numpy as np
import argparse
import matplotlib.pyplot as plt
import shapeExtraction
import imutils
src = None
erosion_size = 0
max_elem = 2
max_kernel_size = 21

def main(image):
    global src
    i = 2000
    src  = cv.imread(r'C:\Users\emile\MAPHIS\datasets\Luton\500\tp_1\0105033010241.jpg', cv.IMREAD_GRAYSCALE)[0:2000,0:2000]


    def removeTrees(src):
        dilatation_size = 4
        dilation_shape = cv.MORPH_RECT
        element = cv.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                        (dilatation_size, dilatation_size))
        dilatation_dst = 255-cv.erode(src, element)
        return dilatation_dst*src

    def segment(src):
        dilatation_size = 1
        dilation_shape = cv.MORPH_RECT
        element = cv.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                        (dilatation_size, dilatation_size))
        dilatation_dst = cv.dilate(src, element)

        outline = 255-dilatation_dst

        dilatation_size = 1
        dilation_shape = cv.MORPH_ELLIPSE
        element = cv.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                        (dilatation_size, dilatation_size))
        dilatation_outline = cv.dilate(outline, element)
        
        cleaned = 255-dilatation_outline

        dilatation_size = 1
        dilation_shape = cv.MORPH_ELLIPSE
        element = cv.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                        (dilatation_size, dilatation_size))
        cleanedOutline = cv.erode(cleaned, element)

        _, cleanedOutline = cv.threshold(255-cleanedOutline, 45, 255, cv.THRESH_BINARY)

        imprints = 255-cleanedOutline

        dist = cv.distanceTransform(imprints, cv.DIST_L2, 3,cv.DIST_MASK_PRECISE )
        cv.normalize(dist, dist, 0, 1.0, cv.NORM_MINMAX)

        _, dist = cv.threshold(dist, 0.01, 1.0, cv.THRESH_BINARY_INV)
  
        dist_8u = dist.astype('uint8')
        # Find total markers
        contours = cv.findContours(dist_8u, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        markers = np.zeros(dilatation_dst.shape, dtype=np.int32)
        for i, cnt in enumerate(contours):
            area = cv.contourArea(cnt)

            if area >10000 and area < 1000000:
                #cv.drawContours(markers, contours, i, (np.random.randint(256), np.random.randint(256), np.random.randint(256)), -1)
                pass
            if area<10000:
                cv.drawContours(src, contours, i, (255,255,255), -1) 
        return src, markers

    src, _ = segment(src)
    plt.matshow(src)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Code for Eroding and Dilating tutorial.')
    parser.add_argument('--input', help='Path to input image.', default='LinuxLogo.jpg')
    args = parser.parse_args()
    main(args.input)
