from pathlib import Path
import torch
import torch.nn as nn 
import torch.optim as optim
import torchvision
from datasets import syntheticCity
import argparse
import models
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='Tree Generation')
    parser.add_argument('--batchSize', required=False, type=int, default = 8)
    parser.add_argument('--randomSeed', required=False, type=int, default = 753159)
    parser.add_argument('--savePath', required=False, type=str, default = 'datasets')
    parser.add_argument('--imageSize', required=False, type=int, default = 512)
    parser.add_argument('--epochs', required=False, type=int, default = 3)
    parser.add_argument('--numWorkers', required=False, type=int, default = 2)
    parser.add_argument('--feature', required=False, type=str, default = '')
    parser.add_argument('--process', required=False, type=str, default = 'segment')
    args = parser.parse_args()

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
        ])

    trainSet = syntheticCity(filepath='/datasets/syntheticCities/', train=True, transform=transform)
    trainDataloader = torch.utils.data.DataLoader(trainSet, batch_size=args.batchSize,
                                            shuffle=True, num_workers=args.numWorkers, drop_last=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    modelSegment = models.segmentationModel(ngf=8, ncOut=1)

    if Path(f'city-drawer/saves/{args.feature}SegmentModelStateDict.pth').is_file():
        modelSegment.load_state_dict(torch.load(f'city-drawer/saves/{args.feature}SegmentModelStateDict.pth'))

    modelSegment.to(device)

    optimizer = optim.Adam(modelSegment.unet.parameters(), lr=0.0001)

    criterion = nn.BCELoss()

    for epoch in range(args.epochs):
        running_loss = 0.0
        for i, data in enumerate(trainDataloader):
            inputImage, treesMask, stripesMask = data[0].float().to(device), data[1].float().to(device), data[2].float().to(device)
            maskDict = {'trees':treesMask, 'stripes':stripesMask}
            optimizer.zero_grad()
            output = modelSegment(inputImage)
            
            loss = criterion(output, (stripesMask)*inputImage)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print(f'[{i}] / [{int(3600/args.batchSize)}] --> Item loss = {loss.item():.4f}')

            if i%100==0:
                plt.imshow(output[0,0].detach().cpu())
                plt.title(f'Segmented Image')
                plt.show()

        torch.save(modelSegment.state_dict(), f'city-drawer/saves/{args.feature}{args.process.capitalize()}ModelStateDict.pth')

if __name__ == '__main__':
    main()