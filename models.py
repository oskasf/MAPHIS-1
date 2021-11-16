from torch.nn import Module, Upsample
from torch import Tensor, where
from CRAFT.craft_utils import getDetBoxes, adjustResultCoordinates
from CRAFT.craft import CRAFT
from collections import OrderedDict
from pathlib import Path
from torch import device, load
from CRAFT.refinenet import RefineNet
from torch.nn import Module, ModuleDict, Conv2d, ConvTranspose2d, Sigmoid, LeakyReLU, parameter, MaxPool2d, Linear
from torch import cat, linspace, meshgrid, arange, cos, sin, exp, linalg, Tensor, flatten
import math
import matplotlib.pyplot as plt

from torchvision import models
from torchvision.models.vgg import model_urls

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

class labelExtractor(Module):
    def __init__(self, savedPathDetection:Path, savedPathRefiner:Path, cudaDevice:device, textThreshold:float, linkThreshold:float, lowText:float) -> None:
        super().__init__()
        self.detectionModel = CRAFT()
        print(f'Loading weights from checkpoint {savedPathDetection}')   
        self.detectionModel.load_state_dict(copyStateDict(load(savedPathDetection)))
        self.detectionModel.to(cudaDevice)
        self.detectionModel.eval()

        self.refinerModel = RefineNet()
        print(f'Loading weights of refiner from checkpoint ({savedPathRefiner})')
        self.refinerModel.load_state_dict(copyStateDict(load(savedPathRefiner)))
        self.refinerModel.to(cudaDevice)
        self.refinerModel.eval()

        self.textThreshold = textThreshold
        self.linkThreshold = linkThreshold
        self.lowText = lowText

        self.Upsample = Upsample(scale_factor=2)

    def forward(self, thumbnail:Tensor) -> list:
        y, feature = self.detectionModel(thumbnail)
        # make score and link map
        y_refiner = self.refinerModel(y, feature)
        y_ = y_refiner[:,:,:,0].unsqueeze(0)
        # Post-processing
        boxes, _ = getDetBoxes(y[0,:,:,0].cpu().data.numpy(), y_refiner[0,:,:,0].cpu().data.numpy(), self.textThreshold, self.linkThreshold, self.lowText, False)
        # coordinate adjustment
        boxes = adjustResultCoordinates(boxes, 1, 1)
        return boxes, where(self.Upsample(y_)>0.1,1,0)

class down2d(Module):
    def __init__(self, inChannels:int, outChannels:int, filterSize:int):
        super(down2d, self).__init__()
        self.pooling2d = Conv2d(inChannels,  inChannels, 4, stride=2, padding=1)        
        self.conv1 = Conv2d(inChannels,  outChannels, filterSize, stride=1, padding=int((filterSize - 1) / 2))        
        self.conv2 = Conv2d(outChannels, outChannels, filterSize, stride=1, padding=int((filterSize - 1) / 2))
        self.lRelu = LeakyReLU(negative_slope = 0.1)
           
    def forward(self, x:Tensor) -> Tensor:
        x = self.lRelu(self.pooling2d(x))
        x = self.lRelu(self.conv1(x))
        x = self.lRelu(self.conv2(x))
        return x
    
class up2d(Module):
    def __init__(self, inChannels:int, outChannels:int):
        super(up2d, self).__init__()
        self.unpooling2d = ConvTranspose2d(inChannels, inChannels, 4, stride = 2, padding = 1)
        self.conv1 = Conv2d(inChannels,  outChannels, 3, stride=1, padding=1)
        self.conv2 = Conv2d(2 * outChannels, outChannels, 3, stride=1, padding=1)
        self.lRelu = LeakyReLU(negative_slope = 0.1)
           
    def forward(self, x:Tensor, skpCn:Tensor) -> Tensor:
        x = self.lRelu(self.unpooling2d(x))
        x = self.lRelu(self.conv1(x))
        x = self.lRelu(self.conv2(cat((x, skpCn), 1)))
        return x    

class UNet2d(Module):
    def __init__(self, inChannels:int, outChannels:int, ngf:int, fs:int):
        super(UNet2d, self).__init__()
        # Initialize neural network blocks.
        self.conv1 = Conv2d(inChannels, ngf, fs, stride=1, padding=int((fs - 1) / 2))
        self.conv2 = Conv2d(ngf, ngf, fs, stride=1, padding=int((fs - 1) / 2))
        self.down1 = down2d(ngf, 2*ngf, 5)
        self.down2 = down2d(2*ngf, 4*ngf, 3)
        self.down3 = down2d(4*ngf, 8*ngf, 3)
        self.down4 = down2d(8*ngf, 16*ngf, 3)
        self.down5 = down2d(16*ngf, 32*ngf, 3)
        self.down6 = down2d(32*ngf, 64*ngf, 3)
        self.down7 = down2d(64*ngf, 64*ngf, 3)
        self.up1   = up2d(64*ngf, 64*ngf)
        self.up2   = up2d(64*ngf, 32*ngf)
        self.up3   = up2d(32*ngf, 16*ngf)
        self.up4   = up2d(16*ngf, 8*ngf)
        self.up5   = up2d(8*ngf, 4*ngf)
        self.up6   = up2d(4*ngf, 2*ngf)
        self.up7   = up2d(2*ngf, ngf)
        self.conv3 = Conv2d(ngf, inChannels, 3, stride=1, padding=1)
        self.conv4 = Conv2d(2*inChannels, outChannels, 3, stride=1, padding=1)
        self.lRelu = LeakyReLU(negative_slope=0.1)
        self.sigmoid = Sigmoid()
        
    def forward(self, x):
        s0  = self.lRelu(self.conv1(x))
        s1 = self.lRelu(self.conv2(s0))
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        s5 = self.down4(s4)
        s6 = self.down5(s5)
        s7 = self.down6(s6)
        u0 = self.down7(s7)
        u1 = self.up1(u0, s7)
        u2 = self.up2(u1, s6)
        u3 = self.up3(u2, s5)
        u4 = self.up4(u3, s4)
        u5 = self.up5(u4, s3)
        u6 = self.up6(u5, s2)
        u7 = self.up7(u6, s1)
        y0 = self.lRelu(self.conv3(u7))
        y1 =self.sigmoid(self.conv4(cat((y0, x), 1)))
        return y1

class segmentationModel(Module):
    def __init__(self, nc=1, nGaborFilters=64, ngf=64, ncOut=2, supportSizes=[5,7,9,11]):
        super(segmentationModel, self).__init__()
        self.name = 'U_GEN'
        self.ngf = ngf
        self.supportSizes = supportSizes
        self.gaborFilters = ModuleDict({f'{supportSize}': Conv2d(nc, int(nGaborFilters/len(supportSizes)), supportSize, stride = 1, padding=int((supportSize-1)/2), padding_mode='reflect'  ) for supportSize in supportSizes})
        
        for param in self.gaborFilters.parameters():
            param.requires_grad = False
        self.setGaborfiltersValues()       
        
        self.unet = UNet2d(nGaborFilters, ncOut, ngf, 5)
        
    def setGaborfiltersValues(self, thetaRange = 180):
        thetas = linspace(0, thetaRange, int(self.ngf/len(self.supportSizes)))
        for supportSize in self.supportSizes:
            filters = gaborFilters(supportSize)
            for indextheta, theta in enumerate(thetas):
                self.gaborFilters[f'{supportSize}'].weight[indextheta][0] = parameter.Parameter(filters.getFilter(theta), requires_grad=False)

    def forward(self, x):
        c5  = self.gaborFilters['5'](x)
        c7  = self.gaborFilters['7'](x)
        c9  = self.gaborFilters['9'](x)
        c11 = self.gaborFilters['11'](x)
        y = cat((c5,c7,c9,c11),1)
        z = self.unet(y)
        return z

class gaborFilters():
    def __init__(self, supportSize):
        self.gridX, self.gridY = meshgrid(arange(-math.floor(supportSize/2),math.ceil(supportSize/2)), arange(-math.floor(supportSize/2),math.ceil(supportSize/2)))
        self.frequency = 1/8
        self.sigma = 3

    def getFilter(self, theta):
        Filter = cos(2*3.1415*self.frequency*(self.gridX*cos(theta) + self.gridY*sin(theta)))*exp(-(self.gridX*self.gridX+self.gridY*self.gridY)/(2*self.sigma*self.sigma))
        return Filter/linalg.norm(Filter)


class tilesClassifier(Module):
    def __init__(self, inChannels:int, outClasses:int, ngf:int, fs:int):
        super(tilesClassifier, self).__init__()
        # Initialize neural network blocks.
        self.conv1 = Conv2d(inChannels, ngf, fs, stride=1, padding=int((fs - 1) / 2))
        self.conv2 = Conv2d(ngf, ngf, fs, stride=1, padding=int((fs - 1) / 2))
        self.down1 = down2d(ngf, 2*ngf, 5)
        self.down2 = down2d(2*ngf, 4*ngf, 3)
        self.down3 = down2d(4*ngf, 8*ngf, 3)
        self.down4 = down2d(8*ngf, 16*ngf, 3)
        self.down5 = down2d(16*ngf, 32*ngf, 3)
        self.down6 = down2d(32*ngf, 64*ngf, 3)
        self.down7 = down2d(64*ngf, 64*ngf, 3)
        self.down8 = down2d(64*ngf, 64*ngf, 3)
        self.fc1 = Linear(256,128)
        self.fc2 = Linear(128,32)
        self.fc3 = Linear(32,5)
        self.lRelu = LeakyReLU(negative_slope=0.1)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        s0  = self.lRelu(self.conv1(x))
        s1 = self.lRelu(self.conv2(s0))
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        s5 = self.down4(s4)
        s6 = self.down5(s5)
        s7 = self.down6(s6)
        s8 = self.down7(s7)
        u0 = self.down8(s8)
        y0 = flatten(u0, 1)
        y1 = self.lRelu(self.fc1(y0))
        y2 = self.lRelu(self.fc2(y1))
        z = self.fc3(y2)
        return z
