"""
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import os
import time
import argparse

from numpy.core.fromnumeric import shape

import torch
import torch.utils.data


import datasets
from torch.utils.data import DataLoader

import glob

import funcUtils
import matplotlib.pyplot as plt
import numpy as np
from models import labelExtractor
from pathlib import Path

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--savedPathDetection', default='CRAFT/weights/craft_mlt_25k.pth', type=str, help='pretrained model for DETECTION')
parser.add_argument('--savedPathRefiner', default='CRAFT/weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model for detection')
parser.add_argument('--textThreshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--lowText', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--linkThreshold', default=0.7, type=float, help='link confidence threshold')
parser.add_argument('--canvas_size', default=512, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.0, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=True, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--cityKey', default='0', type=str, help='Identifying key of the city of interest', required=True)
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--datasetPath', type=str, default = 'datasets/cities', required=False)
args = parser.parse_args()

def main():
    device = torch.device('cuda:0')
    labelExtractorModel = labelExtractor(args.savedPathDetection, args.savedPathRefiner, device, args.textThreshold, args.linkThreshold, args.lowText)

    # load data
    cityName = funcUtils.matchKeyToName('resources/cityKey.json', args.cityKey)

    datasetPath = Path(args.datasetPath)

    cityPath = glob.glob(f'{args.datasetPath }/{cityName}/*/*/')[0]

    saveFolderPath = Path(cityPath) 

    transform = datasets.unfold()
    mapDataset = datasets.Maps(cityPath, transform=transform, fileFormat='jpg')
    mapDataloader = DataLoader(mapDataset, args.batchSize, shuffle=False, num_workers=args.workers)

    sample = mapDataset.__getitem__(0)

    data=next(iter(mapDataloader))
    maskOnly   = np.zeros((7590,11400))
    map = data['map']
    tiledMap = data['tiledMap']
    mapName = data['mapName'][0].split(".")[0]
    mD = data['metaData']
    w_b = mD['west_bound']
    n_b = mD['north_bound']
    x_diff = mD['x_diff']
    y_diff = mD['y_diff']
    for rowIndex in range(transform.hRatio):
        print(f'{rowIndex} / {transform.hRatio}')
        for colIndex in range(transform.wRatio):
            tileIndex = rowIndex*transform.wRatio+colIndex
            tileHindexMap = rowIndex*transform.stride[0]
            tileWindexMap = colIndex*transform.stride[1]
            thumbnail = torch.cat(3*[tiledMap[:,tileIndex]]).unsqueeze(0).cuda(device)
            bBoxes = labelExtractorModel(thumbnail)
            for bBox in bBoxes:
                minW = int(min(bBox, key=lambda x : x[0])[0])
                maxW = int(max(bBox, key=lambda x : x[0])[0])
                minH = int(min(bBox, key=lambda x : x[1])[1])
                maxH = int(max(bBox, key=lambda x : x[1])[1])

                W = maxW - minW
                H = maxH - minH

                x = w_b+(minW + tileWindexMap)*x_diff
                y = n_b+(minH + tileHindexMap)*y_diff
                maskOnly[tileHindexMap+minH:tileHindexMap+minH+H, minW+tileWindexMap:minW+tileWindexMap+W ] = 1
    
    plt.imshow(maskOnly)
    plt.show()
    plt.imshow(maskOnly+(1-maskOnly)*(map[0,0,0].detach().numpy()))
    plt.show()

if __name__ == '__main__':
    main()
