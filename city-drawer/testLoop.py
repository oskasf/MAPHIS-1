import torch
import datasets
import argparse
import models
import matplotlib.pyplot as plt
import numpy as np 
from pathlib import Path
import funcUtils

def main(args):    
    Path('segmentedMaps').mkdir(parents=True, exist_ok=True)
    
    transform = datasets.unfold()
    trainSet = datasets.Maps(filepath=args.datasetPath, fileFormat=args.fileFormat, transform=transform)
    trainDataloader = torch.utils.data.DataLoader(trainSet, batch_size=args.batchSize,
                                            shuffle=True, num_workers=args.numWorkers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    modelSegment = models.segmentationModel(ngf=8, ncOut=1)
    
    modelSegment.load_state_dict(torch.load(f'saves/{args.feature}SegmentModelStateDict.pth'))

    modelSegment.to(device)

    for i, data in enumerate(trainDataloader):
        print(f'Map {i} / {len(trainDataloader)}')
        map = data['map'].float().to(device)
        segmented = np.zeros((7680,11520))

        kS = 256
        nRows = 30
        with torch.no_grad():
            for i in range(nRows):
                print(f'Row {i} / {nRows}')
                for j in range(45):
                    thumbnail = map[:,:,kS*i:kS*(i+1), kS*j:kS*(j+1)]

                    segmented[kS*i:kS*(i+1), kS*j:kS*(j+1)] = modelSegment(thumbnail).detach().cpu()

                if i%10==0:
                    plt.imshow(segmented)
                    plt.title('segmented')
                    plt.show()

        np.save(f'segmentedMaps/{data["mapName"][0].split(".")[0]}_segmented.npy', segmented)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tree Generation')
    parser.add_argument('--batchSize', required=False, type=int, default = 1)
    parser.add_argument('--datasetPath', required=False, type=str, default = f'C:/Users/hx21262/MAPHIS/datasets')
    parser.add_argument('--fileFormat', required=False, type=str, default = 'jpg')
    parser.add_argument('--feature', required=False, type=str, default = 'trees')
    parser.add_argument('--cityName', required=False, type=str, default = 'Luton')
    args = parser.parse_args()

    main(args)

