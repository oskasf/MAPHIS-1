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
    parser.add_argument('--classifType', type=str, required=False, default='labels')
    parser.add_argument('--datasetPath', type=str, required=False, default=r'C:\Users\hx21262\MAPHIS\datasets')
    parser.add_argument('--cityKey', type=str, required=False, default='36')
    parser.add_argument('--tileFileFormat', type=str, required=False, default='.jpg')
    args = parser.parse_args()
    
    cityName = matchKeyToName(f'{args.datasetPath}/cityKey.json', args.cityKey)

    datasetPath = Path(args.datasetPath)

    print(f'Classification Type : {args.classifType}')
    if args.classifType.lower() == 'labels':
        '''defaultFeatureList = ['manhole','lamppost', 'stone', 'chimney', 'chy', 'hotel', 
                            'church', 'workshop', 'firepost', 'river', 'school', 'barrack', 
                            'workhouse', 'market', 'chapel', 'bank', 'pub', 'public house', 'hotel', 
                            'inn', 'bath', 'theatre', 'police', 'wharf', 'yard', 'green', 'park', 'quarry']'''
        from interactiveWindowLabels import Application

    elif args.classifType.lower() == 'tiles':
        '''defaultFeatureList =['rich residential neighborhood', 'poor residential neighborhood', 'industrial district',
                               'peri-urban district',  'farm and forest']'''
        from interactiveWindowTiles import Application

    elif args.classifType.lower() == 'contours':
        #defaultFeatureList = ['interesting','not interesting', 'tree', 'factory', 'villa']
        from interactiveWindowContours import Application
        
    else:
        raise ValueError ("Has to be contours, tiles or labels")

    root = tk.Tk()
    app = Application(root, cityName, datasetPath, args.tileFileFormat)
    root.mainloop()

if __name__=='__main__':
    main()