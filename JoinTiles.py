import json
from datasets.datasetsFunctions import Maps, matchKeyToName, pad
import argparse
import numpy as np
from datasets.datasetsFunctions import Tiles
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

from typing import Tuple


def main():
    parser =argparse.ArgumentParser(usage ='Argument Parser for tiling maps ')
    parser.add_argument('--datasetPath', type=str, required=False, default=r'C:\Users\hx21262\MAPHIS\datasets')
    parser.add_argument('--cityKey', type=str, required=False, default='36')
    parser.add_argument('--savedPathDetection', default='CRAFT/weights/craft_mlt_25k.pth', type=str, help='pretrained model for DETECTION')
    parser.add_argument('--savedPathRefiner', default='CRAFT/weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model for detection')
    parser.add_argument('--textThreshold', default=0.7, type=float, help='text confidence threshold')
    parser.add_argument('--lowText', default=0.4, type=float, help='text low-bound score')
    parser.add_argument('--linkThreshold', default=0.7, type=float, help='link confidence threshold')
    parser.add_argument('--canvas_size', default=512, type=int, help='image size for inference')
    parser.add_argument('--mag_ratio', default=1.0, type=float, help='image magnification ratio')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--mapFileExtension', type=str, default='.jpg', required=False)
    parser.add_argument('--fromCoordinates', type=bool, default=True, required=False)

    args = parser.parse_args()

    cityName = matchKeyToName(f'{args.datasetPath}/cityKey.json', args.cityKey)
    allTilesPaths = list(Path(f'{args.datasetPath}/cities/{cityName}').glob(f'*/*/*{args.mapFileExtension}'))


    minW = -22691.4288240789
    minH = 40234.5927969697
    wSpace = 482.81
    hSpace = -321.86
    canvas = np.ones((7590*5,11400*3,3), np.uint8)*128
    '''for tilePath in allTilesPaths:
        print(f'Processing Tile {tilePath.stem}')
        tilesDataset = Tiles(Path(args.datasetPath), cityName, mapName=tilePath.stem, fromCoordinates=args.fromCoordinates)
        westBoundary, northBoundary = tilesDataset.boundaries['westBoundary'], tilesDataset.boundaries['northBoundary']
        i, j =int((westBoundary-minW)/wSpace), 4 + (int((northBoundary-minH)/hSpace))
        t = np.load(f'datasets/coloredMaps/Luton/{tilePath.stem}.npy')
        canvas[7590*j:7590*(j+1), 11400*i:11400*(i+1),:] = t
      
    # creating image object of
    # above array
    data = Image.fromarray(canvas)
      
    # saving the final output 
    # as a PNG file
    data.save('Luton_segmented.jpg')'''
    canvas = np.ones((7590*5,11400*3), np.bool8)
    for tilePath in allTilesPaths:
        image = np.zeros((7904,11600))
        print(f'Processing Tile {tilePath.stem}')
        map = np.asarray(Image.open(f'datasets/cities/Luton/500/tp_1/{tilePath.stem}.jpg'))
        tilesDataset = Tiles(Path(args.datasetPath), cityName, mapName=tilePath.stem, fromCoordinates=args.fromCoordinates)
        paddedMap = np.pad(map, ((157,157),(100,100)))
        Labels = json.load(open(f'datasets/labels/Luton/{tilePath.stem}.json'))['labels']
        for labelIndex, label in Labels.items():
            H, W = label['H'], label['W']
            xTile, yTile = label['xTile'], label['yTile']
            image[yTile:yTile+H, xTile:xTile+W] = paddedMap[yTile:yTile+H, xTile:xTile+W ]
        t =image[100:7590+100,157:157+11400]
        
        westBoundary, northBoundary = tilesDataset.boundaries['westBoundary'], tilesDataset.boundaries['northBoundary']
        i, j =int((westBoundary-minW)/wSpace), 4 + (int((northBoundary-minH)/hSpace))
        canvas[7590*j:7590*(j+1), 11400*i:11400*(i+1)] = t
      
    # creating image object of
    # above array
    data = Image.fromarray(canvas)
      
    # saving the final output 
    # as a PNG file
    data.save('Luton_labels.jpg')


if __name__=='__main__':
    main()

    
    

