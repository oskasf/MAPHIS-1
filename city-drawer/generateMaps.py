import numpy as np
import math
import cv2
import random
from scipy import ndimage
from skimage.draw import line, disk, ellipse_perimeter, circle_perimeter, rectangle_perimeter, rectangle
from pathlib import Path
import matplotlib.pyplot as plt
import glob
import argparse
from pyprojroot import here

nLinesMax = 8

pSmall  = 0.15
pMedium = 0.15
pLarge  = 0.3
pHuge = 0.4

margin = 5
assert(pSmall + pMedium + pLarge + pHuge == 1)

def crop(mat:float, margin:int, sizeImg:int, center=True) -> float :
    if center:
        return mat[margin:margin+sizeImg,margin:margin+sizeImg]
    else:
        raise NotImplementedError ("Non-centered Crops are not implemented")

def generateStripePattern(sizeImg:int) -> float:
    enclosingSquareLength = int(sizeImg*math.sqrt(2))
    lines = np.ones((int(enclosingSquareLength),int(enclosingSquareLength)), dtype=float)
    i=1
    while i < enclosingSquareLength:
        for j in [i-1,i]:
            rr, cc = line(i,0,i,enclosingSquareLength-1)
            lines[rr, cc] = 0
        rr, cc = line(i,0,i,enclosingSquareLength-1)
        lines[rr, cc] = 0
        i += 7 
    rotationAngle = random.randint(20,90-20) + random.randint(0,1)*90
    rotatedImage = ndimage.rotate(lines, rotationAngle, reshape=True)
    toCrop = np.shape(rotatedImage)[0]-sizeImg
    return rotatedImage[int(toCrop/2):int(toCrop/2)+sizeImg, int(toCrop/2):int(toCrop/2)+sizeImg]

def generate_ellipsoid(maxLength):
    radiusX = random.randint(int(maxLength/4), int(maxLength/3))
    radiusY = random.randint(int(maxLength/4), int(maxLength/3))
    centerX = random.randint(radiusX, maxLength-radiusX )
    centerY = random.randint(radiusY, maxLength-radiusY )
    rr, cc   = ellipse_perimeter(centerX,centerY, radiusX, radiusY)
    return rr, cc
    
def generate_circle(maxLength):
    radius = random.randint(int(maxLength/4), int(maxLength/3))
    centerX = random.randint(radius, maxLength-radius )
    centerY = random.randint(radius, maxLength-radius )
    rr, cc   = circle_perimeter(centerX,centerY, radius)
    return rr, cc

def generate_rectangle(maxLength):
    extent_x = random.randint(int(maxLength/4), int(maxLength/3))
    extent_y = random.randint(int(maxLength/4), int(maxLength/3))      
    start_x = random.randint(extent_x, maxLength-extent_x)
    start_y = random.randint(extent_y, maxLength-extent_y)        
    start = (start_x, start_y)
    extent = (extent_x, extent_y)
    rr, cc = rectangle_perimeter(start, extent=extent)
    return rr, cc

def generateThickRectangle(maxRotatedLength):
    shapeLength = random.randint(int((maxRotatedLength-1)*0.5), maxRotatedLength-1)
    mask = np.ones((shapeLength,shapeLength))
    margin = 5
    mask[0:margin,:] = 0
    mask[shapeLength-margin:,:] = 0
    mask[:,0:margin] = 0
    mask[:,shapeLength-margin:] = 0
    pattern = generateStripePattern(shapeLength)
    rotationAngle = random.randint(0,180)
    rotatedImage = ndimage.rotate(pattern*mask, rotationAngle, reshape=True, mode='constant', cval=1)
    rotatedMaskSegment = ndimage.rotate(mask, rotationAngle, reshape=True, mode='constant', cval=0)
    return rotatedImage, rotatedMaskSegment

def generateTree(patternDict):
    randIndexPattern  = random.randint(0,len(patternDict)-1)
    pattern = patternDict[f'{randIndexPattern}']['pattern']
    mask = patternDict[f'{randIndexPattern}']['mask']
    rotationAngle = random.randint(0,180)
    rotatedImage = ndimage.rotate(pattern, rotationAngle, reshape=True, mode='constant', cval=1)
    rotatedMask = ndimage.rotate(mask, rotationAngle, reshape=True, mode='constant', cval=0)
    return rotatedImage, rotatedMask

def allocateImgsTodict(dictPattern, pattern, mask, count):
    dictPattern[f'{count}'] = {'pattern':pattern, 'mask':mask}
    return dictPattern, count+1

def getPatterns(datasetPath):
    patternsDict = {}
    smallPatterns  = {}
    mediumPatterns = {}
    largePatterns  = {}
    hugePatterns  = {}
    countSmall  = 0
    countMedium = 0
    countLarge  = 0
    countHuge = 0
    for i in range(10):
        pattern = cv2.imread(f'{datasetPath}/arbre{i}.jpg', cv2.IMREAD_GRAYSCALE)
        mask = np.zeros(np.shape(pattern))
        rr, cc = disk((int(np.shape(pattern)[0])/2, int(np.shape(pattern)[1])/2), int(min(np.shape(pattern)))/2, shape=np.shape(mask))
        mask[rr, cc] = 1
        if max(np.shape(pattern))*math.sqrt(2)<64:
            smallPatterns, countSmall = allocateImgsTodict(smallPatterns, pattern, mask, countSmall)
        elif 64<max(np.shape(pattern))*math.sqrt(2)<128:
            mediumPatterns, countMedium = allocateImgsTodict(mediumPatterns, pattern, mask, countMedium)
        elif 128<max(np.shape(pattern))*math.sqrt(2)<256:
            largePatterns, countLarge = allocateImgsTodict(largePatterns, pattern, mask, countLarge)
        else:
            countHuge+=1

    assert(countSmall+countMedium+countLarge+countHuge == len(glob.glob(f'{datasetPath}/*.jpg')))
    patternsDict = {'64':smallPatterns, '128':mediumPatterns, '256':largePatterns, '512':hugePatterns}
    return patternsDict

def fillThumbnail(thumbnailSize, pattern, mask, boundRowLow, boundRowHigh,boundColLow, boundColHigh, imageToFill, maskToFill):
    thumbnail = np.ones((thumbnailSize,thumbnailSize))
    maskToReturn =  np.zeros((thumbnailSize,thumbnailSize))
    posX = random.randint(0, thumbnailSize-np.shape(pattern)[0])
    posY = random.randint(0, thumbnailSize-np.shape(pattern)[1])
    thumbnail[posX:posX+np.shape(pattern)[0], posY:posY+np.shape(pattern)[1]] *= pattern
    maskToReturn[posX:posX+np.shape(mask)[0], posY:posY+np.shape(mask)[1]] += mask
    imageToFill[boundRowLow:boundRowHigh, boundColLow:boundColHigh] *= thumbnail
    maskToFill[boundRowLow:boundRowHigh, boundColLow:boundColHigh] += maskToReturn
    return imageToFill, maskToFill

def generateTreeOrStripe(tbSizeVar, patternsDict, boundRowLow, boundRowHigh,boundColLow, boundColHigh, image, maskTrees, maskStripes, pTree=0.5):
    patternVar = random.choices([0,1], [pTree,1-pTree])[0]
    if patternVar ==0:
        pattern, mask = generateTree(patternsDict)
        image, maskTrees = fillThumbnail(tbSizeVar, pattern, mask, boundRowLow, boundRowHigh,boundColLow, boundColHigh, image, maskTrees)
    else:
        pattern, mask = generateThickRectangle(int(tbSizeVar/math.sqrt(2)))
        image, maskStripes = fillThumbnail(tbSizeVar, pattern, mask, boundRowLow, boundRowHigh,boundColLow, boundColHigh, image, maskStripes)
    return image, maskTrees, maskStripes

def generateBlockOfFlats(sizeImg):
    margin = 3
    enclosingSquareLength = int(sizeImg*math.sqrt(2))
    mask = np.zeros((enclosingSquareLength,enclosingSquareLength))
    band = np.ones((enclosingSquareLength,enclosingSquareLength))
    middle = int(enclosingSquareLength/2)
    width = random.choices([64,128], [0.5,0.5])[0]
    widthMargin = enclosingSquareLength%width
    band[:, middle-width-margin:middle+width+margin] = 0
    mask[margin*2:widthMargin-margin, middle-width+margin:middle+width-margin] = 1
    for i in range(enclosingSquareLength//width):
        mask[widthMargin+width*i+margin:widthMargin+width*(i+1)-margin, middle-width+margin:middle+width-margin] = 1

    rotationAngle = random.randint(0,180)
    rotatedBand = ndimage.rotate(band, rotationAngle, reshape=True, mode='constant', cval=1)
    rotatedImage = ndimage.rotate(mask*generateStripePattern(enclosingSquareLength), rotationAngle, reshape=True, mode='constant', cval=1)
    rotatedMask = ndimage.rotate(mask, rotationAngle, reshape=True, mode='constant', cval=0)
    cropMargin = int((enclosingSquareLength-sizeImg)/2)
    return crop(rotatedImage+rotatedBand, cropMargin, sizeImg), crop(rotatedMask, cropMargin, sizeImg), crop(rotatedBand, cropMargin, sizeImg)

def generateTreesAndMask(patternsDict, sizeImg=512):
    image = np.ones((sizeImg,sizeImg))
    maskTrees = np.zeros((sizeImg,sizeImg))
    maskStripes = np.zeros((sizeImg,sizeImg))
    tbSizeVar = random.choices([256,sizeImg], [pSmall+pMedium+pLarge, pHuge])[0]
    if tbSizeVar == sizeImg:   
        blockOfFlats = random.choices([0,1], [0.5,0.5])[0]
        for indexRow256 in range(2):
            for indexCol256 in range(2):
                boundRowLow  = indexRow256 * 256
                boundRowHigh = boundRowLow + 256
                boundColLow  = indexCol256 * 256
                boundColHigh = boundColLow + 256
                tbSizeVar = random.choices([128,256], [pSmall+pMedium, pLarge])[0]
                if tbSizeVar == 256:   
                    image, maskTrees, maskStripes  = generateTreeOrStripe(256, patternsDict[f'{256}'], boundRowLow, boundRowHigh,boundColLow, boundColHigh, image, maskTrees, maskStripes)  

                else:
                    for indexRow128 in range(2):
                        for indexCol128 in range(2):
                            boundRowLow  = indexRow256 * 256 + indexRow128 * 128
                            boundRowHigh = boundRowLow + 128
                            boundColLow  = indexCol256 *256 + indexCol128 *128
                            boundColHigh = boundColLow + 128
                            tbSizeVar = random.choices([64,128], [pSmall+pLarge, pMedium+pLarge])[0]
                            if tbSizeVar == 128:
                                image, maskTrees, maskStripes  = generateTreeOrStripe(128, patternsDict[f'{128}'], boundRowLow, boundRowHigh,boundColLow, boundColHigh, image, maskTrees, maskStripes)  
                                
                            else:
                                for indexRow64 in range(2):
                                    for indexCol64 in range(2):
                                        boundRowLow  = indexRow256 * 256 + indexRow128 * 128 + indexRow64 * 64
                                        boundRowHigh = boundRowLow + 64
                                        boundColLow  = indexCol256 *256 + indexCol128 *128 + indexCol64 *64
                                        boundColHigh = boundColLow + 64
                                        image, maskTrees, maskStripes = generateTreeOrStripe(64, patternsDict[f'{64}'], boundRowLow, boundRowHigh,boundColLow, boundColHigh, image, maskTrees, maskStripes)           
        if blockOfFlats == 0:
            imageBloc, maskTrees, maskBloc  = generateTreeOrStripe(sizeImg, patternsDict[f'{sizeImg}'], 0, sizeImg,0, sizeImg, np.ones((sizeImg,sizeImg)), maskTrees, np.zeros((sizeImg,sizeImg)), pTree=0) 
            rim = (imageBloc*(maskBloc-1)+(1-maskBloc))
            image = image * (1- maskBloc) + maskBloc*imageBloc - rim
            maskStripes = ( maskStripes * (1-maskBloc) + maskBloc) * (1-rim)
            maskTrees *= 1-maskBloc
        else:
            imageBloc, maskBloc , band  = generateBlockOfFlats(sizeImg)
            image = image*band + imageBloc* (1-band)
            maskStripes  = maskStripes*band + maskBloc
            maskTrees  = maskTrees*band 
    else:
        for indexRow256 in range(2):
            for indexCol256 in range(2):
                boundRowLow  = indexRow256 * 256
                boundRowHigh = boundRowLow + 256
                boundColLow  = indexCol256 * 256
                boundColHigh = boundColLow + 256
                tbSizeVar = random.choices([128,256], [pSmall+pMedium, pLarge])[0]
                if tbSizeVar == 256:   
                    image, maskTrees, maskStripes  = generateTreeOrStripe(256, patternsDict[f'{256}'], boundRowLow, boundRowHigh,boundColLow, boundColHigh, image, maskTrees, maskStripes)  

                else:
                    for indexRow128 in range(2):
                        for indexCol128 in range(2):
                            boundRowLow  = indexRow256 * 256 + indexRow128 * 128
                            boundRowHigh = boundRowLow + 128
                            boundColLow  = indexCol256 *256 + indexCol128 *128
                            boundColHigh = boundColLow + 128
                            tbSizeVar = random.choices([64,128], [pSmall+pLarge, pMedium+pLarge])[0]
                            if tbSizeVar == 128:
                                image, maskTrees, maskStripes  = generateTreeOrStripe(128, patternsDict[f'{128}'], boundRowLow, boundRowHigh,boundColLow, boundColHigh, image, maskTrees, maskStripes)  
                                
                            else:
                                for indexRow64 in range(2):
                                    for indexCol64 in range(2):
                                        boundRowLow  = indexRow256 * 256 + indexRow128 * 128 + indexRow64 * 64
                                        boundRowHigh = boundRowLow + 64
                                        boundColLow  = indexCol256 *256 + indexCol128 *128 + indexCol64 *64
                                        boundColHigh = boundColLow + 64
                                        image, maskTrees, maskStripes = generateTreeOrStripe(64, patternsDict[f'{64}'], boundRowLow, boundRowHigh,boundColLow, boundColHigh, image, maskTrees, maskStripes)           
            
    lines = np.ones((sizeImg,sizeImg), dtype=np.uint8)
    for i in range(random.randint(0,nLinesMax)):
        if random.randint(0,1) == 0:
            r0, r1 = 0, sizeImg-1
            c0, c1 = random.randint(0,sizeImg-1), random.randint(0,sizeImg-1)
        else:
            c0, c1 = 0,sizeImg-1
            r0, r1 = random.randint(0,sizeImg-1), random.randint(0,sizeImg-1)
        rr, cc = line(r0,c0,r1,c1)
        lines[rr, cc] = 0

    _, image = cv2.threshold(image, 0.2, 1, cv2.THRESH_BINARY)
    return lines*image, maskTrees, maskStripes

def main(args):
    patternsDict = getPatterns(args.datasetPath)
    #random.seed(args.randomSeed)
    counter = len(glob.glob(f'{args.savePath}/image*'))
    for i in range(args.nSamples):
        image, maskTrees, maskStripes = generateTreesAndMask(patternsDict)
        if args.treatment == 'show':
            plt.matshow(image)
            plt.show()
            plt.matshow(maskStripes + maskTrees)
            plt.show()

        elif args.treatment == 'save':
            Path(f'{args.savePath}').mkdir(parents=True ,exist_ok=True)
            np.save(f'{args.savePath}/image_{counter}', image)
            np.save(f'{args.savePath}/maskTrees_{counter}', maskTrees)
            np.save(f'{args.savePath}/maskStripes_{counter}', maskStripes)
        else:
            raise NotImplementedError ("Can only save or show")

        counter+=1
        if i%int(args.nSamples/10)==0:
            print(f'{i} / {args.nSamples}')

if __name__ == '__main__':
    base_path = str(here())
    parser = argparse.ArgumentParser(description='Tree Generation')
    parser.add_argument('--datasetPath', required=False, type=str, default = base_path / "datasets" / "patterns")
    parser.add_argument('--nSamples', required=False, type=int, default = 4000)
    parser.add_argument('--randomSeed', required=False, type=int, default = 753159)
    parser.add_argument('--savePath', required=False, type=str, default = "datasets" / "syntheticCities")
    parser.add_argument('--imageSize', required=False, type=int, default = 512)
    parser.add_argument('--treatment', required=False, type=str, default='save')
    args = parser.parse_args()

    savePath = Path(args.savePath)
    savePath.mkdir(parents=True, exist_ok=True)

    main(args)
