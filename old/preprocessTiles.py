def main():
    import funcUtils
    import glob
    import datasets
    from torch.utils.data import DataLoader
    import argparse
    from pathlib import Path, PurePath
    import torch
    import csv
    import numpy as np
    from contextlib import contextmanager

    parser = argparse.ArgumentParser(description='CRAFT Text Detection')
    parser.add_argument('--cityKey', default='0', type=str, help='Identifying key of the city of interest', required=True)
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
    parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--datasetPath', type=str, default = 'datasets/', required=False)
    parser.add_argument('--savedFileExtension', type=str, default='.npz', required=False)
    args = parser.parse_args()

    cityName = funcUtils.matchKeyToName('resources/cityZone.txt', args.cityKey)

    datasetPath = glob.glob(args.datasetPath+cityName+'/*/*/')[0]

    '''
    remoteDirPath = r'\\rdsfcifs.acrc.bris.ac.uk\MAPHIS_historical_maps\Data\datasets'

    saveFolderPath = Path(PurePath(remoteDirPath).joinpath(PurePath(cityName)))
    saveFolderPath.mkdir(parents=True, exist_ok=True)
    '''
    saveFolderPath = Path(datasetPath)

    transform = datasets.unfold()
    mapDataset = datasets.Maps(datasetPath, transform=transform)
    mapDataloader = DataLoader(mapDataset, args.batchSize, shuffle=True, num_workers=args.workers)

    with torch.no_grad():
        for data in mapDataloader:
            tiledMap = data['map']
            tileName = data['mapName'][0].split(".")[0]
            mD = data['metaData']
            westBound = mD['west_bound']
            northBound = mD['north_bound']
            xDiff = mD['x_diff']
            yDiff = mD['y_diff']
            header = ['tileIndex', 'westBound', 'northBound', 'xDiff', 'yDiff', 'tileWindexMap', 'tileHindexMap']
            for rowIndex in range(transform.hRatio):
                for colIndex in range(transform.wRatio):
                    tileIndex = rowIndex*transform.wRatio+colIndex
                    tileHindexMap = rowIndex*transform.stride[0]
                    tileWindexMap = colIndex*transform.stride[1]
                    thumbnailList = [tileIndex, westBound.item(), northBound.item(), xDiff.item(), yDiff.item(), tileWindexMap, tileHindexMap]
                    filePath = saveFolderPath / f'{tileName}_{tileIndex}.csv'
                    npzFilePath = saveFolderPath / f'{tileName}_{tileIndex}.npz'
                    saveDict = {}
                    torch.save(saveDict, filePath)

                    with csvFilePath.open('w', newline='') as csvFile:
                        fileWriter = csv.writer(csvFile)
                        fileWriter.writerow(header)
                        fileWriter.writerow(thumbnailList)
                    
                    np.savez(npzFilePath,) 
        
if __name__ == '__main__':
    main()