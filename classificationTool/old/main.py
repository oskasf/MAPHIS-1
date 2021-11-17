# -*- coding: utf-8 -*-

import csv
from pathlib import PurePath


def main():
    
    import tkinter as tk
    from funcUtils import createFolders
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument('--rawFolderName', type=str, default='rawFolder', required=False)
    parser.add_argument('--classifiedFolderName', type=str, default='classifiedFolder', required=False)
    parser.add_argument('--extensionName', type=str, default='zip', required=False)
    parser.add_argument('--classifType', type=str, required = True)
    parser.add_argument('--unzip', type=bool, required=False, default=False)
    args = parser.parse_args()
    
    Path(PurePath(args.classifiedFolderName)).mkdir(parents=True, exist_ok=True)

    if args.classifType.lower() == 'labels':
        defaultFeatureList = ['manhole','lamppost', 'stone', 'chimney', 'chy', 'hotel', 
                            'church', 'workshop', 'firepost', 'river', 'school', 'barrack', 
                            'workhouse', 'market', 'chapel', 'bank', 'pub', 'public house', 'hotel', 
                            'inn', 'bath', 'theatre', 'police', 'wharf', 'yard', 'green', 'park', 'quarry' ]
        from interactiveWindowLabels import Application
        featureListName = 'featureListLabels.csv'

    elif args.classifType.lower() == 'tiles':
        defaultFeatureList = ['rich residential neighborhood', 'poor residential neighborhood', 'industrial district',
                               'peri-urban district',  'farm and forest']
        from interactiveWindowTiles import Application
        featureListName = 'featureListTiles.csv'

    elif args.classifType.lower() == 'contours':
        defaultFeatureList = ['interesting','not interesting', 'tree', 'factory', 'villa']
        from interactiveWindowContours import Application
        featureListName = 'featureListContours.csv'

    else:
        raise ValueError ("Has to be contours, tiles or labels")

    if args.unzip:
        createFolders(args.rawFolderName, args.classifiedFolderName, args.extensionName)
    else:
        print("Folders already created") 
    
    while True:
        cityName = input('Enter City Name:')
        if not Path(PurePath(args.rawFolderName).joinpath(cityName)).is_dir():
            print(f"Wrong city Name : city not in {args.rawFolderName}")
            continue
        else:
            break
    
    ## Check if feature List file exists, creates it if not
    fp = Path(PurePath(args.classifiedFolderName).joinpath(featureListName))
    if not fp.is_file():
        with open(f'{fp}', 'w', newline='') as csvFile:
            fileWriter = csv.writer(csvFile)
            for featureName in defaultFeatureList:
                fileWriter.writerow([featureName])

    root = tk.Tk()
    app = Application(args.classifType.lower(), root, args.rawFolderName, cityName, args.classifiedFolderName)
    
    root.mainloop()


if __name__=='__main__':
    main()