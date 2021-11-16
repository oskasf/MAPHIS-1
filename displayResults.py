import argparse
from datasets.datasetsFunctions import matchKeyToName, openfile
from pathlib import Path
import json
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetPath', required=False, type=str, default='datasets')
    parser.add_argument('--cityKey', required=False, type=str, default = '36')
    parser.add_argument('--classifType', required=False, type=str, default = 'Maps')
    args = parser.parse_args()

    cityName = matchKeyToName(f'{args.datasetPath}/cityKey.json', args.cityKey)
    classifiedTilesPaths = list(Path(f'{args.datasetPath}/classified{args.classifType}/{cityName}').glob('*.json'))
    classes = openfile(Path(f'{args.datasetPath}/classified{args.classifType}/classes.json'), '.json')

    def getMapClassificationDistribution(pathToMap:Path) -> np.float32:
        mapDistribution = np.zeros(len(classes), np.float32)
        print(f'Processing tile {pathToMap.stem}')
        jsonDict = openfile(pathToMap, pathToMap.suffix)
        unClassifiedIndexes = jsonDict['Not Classified']
        classifiedIndexes = jsonDict['Classified']
        totalLength = len(classifiedIndexes)+len(unClassifiedIndexes)
        print(f'classification percentage : { len(classifiedIndexes)/totalLength:.2f}')       
        for classifiedIndex in classifiedIndexes:
            mapDistribution[int(jsonDict[f'{classifiedIndex}'])] += 1
        return mapDistribution/totalLength

    for path in classifiedTilesPaths:
        mapDistribution = getMapClassificationDistribution(path)    
        print(mapDistribution)


if __name__ == '__main__':
    main()