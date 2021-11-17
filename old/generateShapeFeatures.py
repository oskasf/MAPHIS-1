"""
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import time
import argparse

from numpy.core.fromnumeric import shape

import datasets
from torch.utils.data import DataLoader

import glob

import funcUtils

from pathlib import Path, PurePath
import shapeExtraction
import numpy as np
import cv2
import csv
import shutil

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--cityKey', default='0', type=str, help='Identifying key of the city of interest', required=True)
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--datasetPath', type=str, default = 'datasets/', required=False)
parser.add_argument('--savedFileExtension', type=str, default='.npz', required=False)
parser.add_argument('--saveFolderPath', type=str, default = 'results/rawShapes', required=False)
parser.add_argument('--saveContours', type=str2bool, required=True, help='True or False')
args = parser.parse_args()


def main():
    cityName = funcUtils.matchKeyToName('resources/cityZone.txt', args.cityKey)

    datasetPath = glob.glob(args.datasetPath+cityName+'/*/*/')[0]

    saveFolder = Path(PurePath(args.saveFolderPath).joinpath(PurePath(cityName)))
    saveFolder.mkdir(parents=True, exist_ok=True)

    transform = datasets.unfold()
    mapDataset = datasets.Maps(datasetPath, transform=transform)
    mapDataloader = DataLoader(mapDataset, args.batchSize, shuffle=False, num_workers=args.workers)
    
    header = ['x', 'y', 'cityName', 'tileName', 'H', 'W', 'xTile', 'yTile', 'perimeter', 'perimeterApproximation',
            'complexity', 'area', 'circleness', 'rectangleness', 'solidity']
    for i in range(25):
        header.append(f'zernikeDecomposition_{i}')

    print(header)
    for data in mapDataloader:
        tileName = data['mapName'][0].split(".")[0]
        mD = data['metaData']
        w_b = mD['west_bound'].item()
        n_b = mD['north_bound'].item()
        x_diff = mD['x_diff'].item()
        y_diff = mD['y_diff'].item()
        print(f'Beginning Shape extraction')
        t0 = time.time()
        image = cv2.imread(data['tilePath'][0], cv2.IMREAD_GRAYSCALE)
        mask = np.zeros(image.shape[:2], dtype="uint8")
        contours = shapeExtraction.getContours(image)
        indexContour = 0
        print(f'{len(contours)} shapes detected; processing')
        for contour in contours:                
            cv2.drawContours(mask, [contour], -1, 255, -1)
            contourDict = {'cityName':cityName, 'tileName':tileName}                
            shapeDict = shapeExtraction.extractGeometricalProperties(contour, mask)
            contourDict['x'] = w_b + shapeDict['xTile']*x_diff
            contourDict['y'] = n_b + shapeDict['yTile']*y_diff
            for key, value in shapeDict.items():
                contourDict[key] = value
            
            if args.saveContours:
                locSavePath = saveFolder / Path(tileName+'_'+str(indexContour)+args.savedFileExtension)
                dictToSave = {'image':image, 'features':contourDict}
                np.savez(f'{locSavePath}',dictToSave)

            indexContour +=1
            if indexContour % int(len(contours)/10)==0:
                print(f'{indexContour} / {len(contours)}')

        shutil.make_archive(saveFolder / 'rawShapesFiles', 'zip', saveFolder )        
    print(f'Shape detection elapsed Time : {time.time()-t0}')     

if __name__ == '__main__':
    main()
