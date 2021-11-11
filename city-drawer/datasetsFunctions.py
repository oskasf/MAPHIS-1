#!/usr/bin/env python
# coding: utf-8
from pathlib import Path
from torch.utils.data import Dataset
import glob
from PIL import Image
import torch.nn as nn
from torchvision.transforms import ToTensor
import numpy as np
import pandas as pd
import json
from PIL.TiffTags import TAGS

def csv_from_excel(fileName):
    pd.read_excel('./'+fileName).to_csv('./'+fileName[:fileName.index('.')]+'.csv', index=False)
    
def normalise(tensor):
    return (tensor-tensor.min())/(tensor.max()-tensor.min()) 

def identity(tensor):
    return tensor

def getTiffProperties(tiffImage, showDict = False, returnDict=False):    
    meta_dict = {TAGS[key] : tiffImage.tag[key] for key in tiffImage.tag.keys()}
    if showDict:
        for key, value in meta_dict.items():
            print(' %s : %s' % (key, value))
    if returnDict:
        return meta_dict

class syntheticCity(Dataset):
    def __init__(self, datasetPath:Path, train=True, fileFormat='.npy', transform = None):
        filePath = datasetPath / f'syntheticCities'
        self.lenDataset = len(list(filePath.glob(f'maskTrees*{fileFormat}')))
        if train:
            low, high = int(self.lenDataset*0.1), -1
        else:
            low, high = 0, int(self.lenDataset*0.1)

        self.maskTrees   = list(filePath.glob(f'maskTrees*{fileFormat}'))[low:high]
        self.maskStripes = list(filePath.glob(f'maskStripes*{fileFormat}'))[low:high]
        self.images = list(filePath.glob(f'image*{fileFormat}'))[low:high]
        self.transform= transform

    def __len__(self): 
        return len(self.maskTrees)

    def __getitem__(self, index):  
        image = np.load(self.images[index])
        maskTree = np.load(self.maskTrees[index]) 
        maskStripes = np.load(self.maskStripes[index]) 
        if self.transform:
            image, maskTree, maskStripes = self.transform(image), self.transform(maskTree), self.transform(maskStripes)
        return image, maskTree, maskStripes

class Maps(Dataset):
    def __init__(self, datasetPath:Path, cityName:str, fileFormat='.jpg', transform=None):
        self.mapsPath   = list(datasetPath.glob(f'cities/{cityName}/*/*/*{fileFormat}'))
        if fileFormat == '.jpg': 
            self.height = 7590
            self.width  = 11400  
        elif fileFormat == '.tif':
            self.height = getTiffProperties(Image.open(self.mapsPath[0]), returnDict=True)['ImageLength'][0]
            self.width  = getTiffProperties(Image.open(self.mapsPath[0]), returnDict=True)['ImageWidth'][0]
        else:
            raise Exception('Wrong File format : only png and tif accepted')        
        self.fileFormat = fileFormat
        
        self.evaluationData = list(datasetPath.glob(f'cities/{cityName}/*/*/*.csv'))
        projectionData = list(datasetPath.glob(f'cities/{cityName}/*/*/*.prj'))
        tfwData = list(datasetPath.glob(f'cities/{cityName}/*/*/*.tfw'))
        self.datasetPath = datasetPath      
        self.projections = projectionData
        self.tfwData = tfwData
        self.transform = transform

    def __getitem__(self, index):                
        if self.fileFormat == '.tif':
            properties =  getTiffProperties(Image.open(self.mapsPath[index]), returnDict=True)
        else: 
            properties =  'No properties with '+self.fileFormat+' format.'
        map = ToTensor()(Image.open(self.mapsPath[index]))
        projection = open(self.projections[index], 'r').read() 
        metaData = self.extractMetaData(open(self.tfwData[index], 'r').read())
        boundaries = self.getBoundaries(metaData, self.height, self.width)
        sample = {'map': map.unsqueeze_(0),
                  'properties':properties,
                  'projection':projection,
                  'metaData':metaData,
                  'boundaries':boundaries,
                  'tilePath':str(self.mapsPath[index]),
                  'mapName':self.mapsPath[index].name
                  }

        if self.transform:
            sample = self.transform(sample)
        return sample    

    def __len__(self): 
        return len(self.mapsPath)
    
    def extractMetaData(self,tfw_raw_data):
        x_diff = float(tfw_raw_data.split("\n")[0])
        y_diff = float(tfw_raw_data.split("\n")[3])
        west_bound = float(tfw_raw_data.split("\n")[4])
        north_bound = float(tfw_raw_data.split("\n")[5])
        return {'x_diff':x_diff, 'y_diff':y_diff, 'west_bound':west_bound, 'north_bound':north_bound}

    def getBoundaries(self, metaData, imageHeight, imageWidth):
        east_bound = metaData['west_bound'] + (imageWidth - 1) * metaData['x_diff']
        south_bound = metaData['north_bound'] + (imageHeight - 1) * metaData['y_diff']
        return {'west_bound':metaData['west_bound'], 'north_bound':metaData['north_bound'],
                'east_bound':east_bound, 'south_bound':south_bound }

class pad(object):
    def __init__(self, paddingX=188, paddingY=45):
        self.paddingMap = nn.ConstantPad2d((paddingX,paddingX, paddingY,paddingY),1)
        
    def __call__(self, sample):
        return {'map': self.paddingMap(sample['map']),
                'tilePath':sample['tilePath'],
                'properties':sample['properties'],
                'projection':sample['projection'],
                'metaData':sample['metaData'],
                'boundaries':sample['boundaries'],
                'mapName':sample['mapName']
                }

def openfile(filePath, fileExtension):
    if fileExtension =='.npy':
        return np.load(filePath)
    else:
        raise ValueError ('Wrong fileExtension string')