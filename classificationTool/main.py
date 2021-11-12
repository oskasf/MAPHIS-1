# -*- coding: utf-8 -*-
import csv
from pathlib import Path
import tkinter as tk
import argparse
import json

def matchKeyToName(pathToJsonfile:str, key : str):
    cityKeysFile = json.load(open(pathToJsonfile))
    return cityKeysFile[key]['Town']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classifType', type=str, required=False, default='Tiles')
    parser.add_argument('--datasetPath', type=str, required=False, default='C:/Users/hx21262/MAPHIS/datasets')
    parser.add_argument('--cityKey', type=str, required=False, default='36')
    args = parser.parse_args()
    
    cityName = matchKeyToName(f'{args.datasetPath}/cityKey.json', args.cityKey)

    datasetPath = Path(args.datasetPath)

    classifiedFolderPath = Path(f'{args.datasetPath}/classifiedMaps/{cityName}')
    classifiedFolderPath.mkdir(parents=True, exist_ok=True)

    print(f'Classification Type : {args.classifType}')
    if args.classifType.lower() == 'labels':
        defaultFeatureList = ['manhole','lamppost', 'stone', 'chimney', 'chy', 'hotel', 
                            'church', 'workshop', 'firepost', 'river', 'school', 'barrack', 
                            'workhouse', 'market', 'chapel', 'bank', 'pub', 'public house', 'hotel', 
                            'inn', 'bath', 'theatre', 'police', 'wharf', 'yard', 'green', 'park', 'quarry' ]
        from interactiveWindowLabels import Application

    elif args.classifType.lower() == 'tiles':
        defaultFeatureList = ['rich residential neighborhood', 'poor residential neighborhood', 'industrial district',
                               'peri-urban district',  'farm and forest']
        from interactiveWindowTiles import Application

    elif args.classifType.lower() == 'contours':
        defaultFeatureList = ['interesting','not interesting', 'tree', 'factory', 'villa']
        from interactiveWindowContours import Application
        
    else:
        raise ValueError ("Has to be contours, tiles or labels")

    featureListName = f'featureList{args.classifType.capitalize()}.csv'
    
    ## Check if feature List file exists, creates it if not
    fp = Path(f'{args.datasetPath}/classifiedMaps/{featureListName}')
    if not fp.is_file():
        with open(fp, 'w', newline='') as csvFile:
            fileWriter = csv.writer(csvFile)
            for featureName in defaultFeatureList:
                fileWriter.writerow([featureName])

    root = tk.Tk()
    app = Application(root, cityName, datasetPath, classifiedFolderPath)
    root.mainloop()

if __name__=='__main__':
    main()