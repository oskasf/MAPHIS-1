from datasetsFunctions import Maps, pad, matchKeyToName
import argparse
from torch.utils.data import DataLoader
import numpy as np
import glob
import json

def main():
    parser =argparse.ArgumentParser(usage ='Argument Parser for tiling maps ')
    parser.add_argument('--datasetPath', type=str, required=False, default='cities')
    parser.add_argument('--cityKey', type=str, required=False, default='36')
    args = parser.parse_args()

    paddingX, paddingY = 100,157
    kernelSize = 512
    strideX = 50
    strideY = 50
    nCols = int((2*paddingX+11400-kernelSize)/(kernelSize-strideX))
    nRows = int((2*paddingY+7590-kernelSize)/(kernelSize-strideY))
    tilingDict = {'height':2*paddingY+7590 , 'width':2*paddingX+11400, 'kernelSize':kernelSize, 'paddingX':paddingX, 'paddingY':paddingY,'strideX':strideX, 'strideY':strideY, 'nCols':nCols, 'nRows':nRows}
    tilingDict['coordinates'] = {}
    nTiles = 0
    for rowIndex in range(nRows+1):
            yLow  = (kernelSize - strideY)*rowIndex
            yHigh = kernelSize*(rowIndex+1) - strideY*rowIndex
            for colIndex in range(nCols+1):
                xLow  = (kernelSize - strideX)*colIndex
                xHigh = kernelSize*(colIndex+1) - strideX*colIndex
                tilingDict['coordinates'][nTiles] = {'yLow':yLow, 'yHigh':yHigh, 'xLow':xLow, 'xHigh':xHigh}
                nTiles+=1
    with open(f'tilingParameters.json', 'w') as outfile:
        json.dump(tilingDict, outfile)
    
    '''for data in mapDataloader:
        mapName = data['mapName'][0].split(".")[0]
        mD = data['metaData']
        westBoundary = mD['west_bound']
        northBoundary = mD['north_bound']
        xDiff = mD['x_diff']
        yDiff = mD['y_diff']
        map = data['map'][0,0,0]
        tilingDict['westBoundary']  = westBoundary.item()
        tilingDict['northBoundary'] = northBoundary.item()
        tilingDict['xDiff'] = xDiff.item()
        tilingDict['yDiff'] = yDiff.item()
        tilingDict['coordinates'] = {}
        tilesDict = {}
        nTiles = 0
        for rowIndex in range(nRows+1):
            yLow  = (kernelSize - strideY)*rowIndex
            yHigh = kernelSize*(rowIndex+1) - strideY*rowIndex
            for colIndex in range(nCols+1):
                xLow  = (kernelSize - strideX)*colIndex
                xHigh = kernelSize*(colIndex+1) - strideX*colIndex
                #np.save(folderPath + f"{mapName}_{nTiles}", map[yLow:yHigh,xLow:xHigh])
                tilingDict['coordinates'][nTiles] = {'yLow':yLow, 'yHigh':yHigh, 'xLow':xLow, 'xHigh':xHigh}
                nTiles+=1
        with open(f'tilingParameters.json', 'w') as outfile:
            json.dump(tilingDict, outfile)'''

if __name__=='__main__':
    main()


