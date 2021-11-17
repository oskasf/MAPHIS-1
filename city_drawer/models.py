from torch.nn import Module, ModuleDict, Conv2d, ConvTranspose2d, Sigmoid, LeakyReLU, parameter
from torch import cat, linspace, meshgrid, arange, cos, sin, exp, linalg, Tensor
import numpy as np
import math

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
        
    def forward(self, x :Tensor):
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
    def __init__(self, parametersDict:dict):
        super(segmentationModel, self).__init__()
        self.name = 'U_GEN'
        ## Assert that all parameters are here:
        for paramKwd in ['ngf', 'ncIn', 'ncOut', 'nGaborFilters', 'supportSizes']:
            if not parametersDict[paramKwd]:raise KeyError (f'{paramKwd} is missing')
        self.ngf = parametersDict['ngf']
        self.supportSizes = parametersDict['supportSizes']
        self.gaborFilters = ModuleDict({f'{supportSize}': Conv2d(parametersDict['ncIn'], int(parametersDict['nGaborFilters']/len(self.supportSizes)), supportSize, stride = 1, padding=int((supportSize-1)/2), padding_mode='reflect'  ) for supportSize in self.supportSizes})
        
        for param in self.gaborFilters.parameters():
            param.requires_grad = False
        self.setGaborfiltersValues()       
        
        self.unet = UNet2d(parametersDict['nGaborFilters'], parametersDict['ncOut'], self.ngf, 5)
        
    def setGaborfiltersValues(self, thetaRange = 180):
        thetas = linspace(0, thetaRange, int(self.ngf/len(self.supportSizes)))
        for supportSize in self.supportSizes:
            filters = gaborFilters(supportSize)
            for indextheta, theta in enumerate(thetas):
                self.gaborFilters[f'{supportSize}'].weight[indextheta][0] = parameter.Parameter(filters.getFilter(theta), requires_grad=False)

    def forward(self, x:Tensor):
        c5  = self.gaborFilters['5'](x)
        c7  = self.gaborFilters['7'](x)
        c9  = self.gaborFilters['9'](x)
        c11 = self.gaborFilters['11'](x)
        y = cat((c5,c7,c9,c11),1)
        z = self.unet(y)
        return z

class gaborFilters():
    def __init__(self, supportSize:int):
        self.gridX, self.gridY = meshgrid(arange(-math.floor(supportSize/2),math.ceil(supportSize/2)), arange(-math.floor(supportSize/2),math.ceil(supportSize/2)))
        self.frequency = 1/8
        self.sigma = 3

    def getFilter(self, theta:int) -> np.float32:
        Filter = cos(2*3.1415*self.frequency*(self.gridX*cos(theta) + self.gridY*sin(theta)))*exp(-(self.gridX*self.gridX+self.gridY*self.gridY)/(2*self.sigma*self.sigma))
        return Filter/linalg.norm(Filter)

