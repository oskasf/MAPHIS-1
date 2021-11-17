import numpy as np
import matplotlib.pyplot as plt
import json

'''plt.imshow(np.load('datasets/coloredMaps/Luton/0105033050201.npy'))
plt.show()'''

classificationfile = json.load(open('datasets/classifiedMaps/Luton/0105033050201.json'))
colorDict = {'0':(128,128,128),'1':(255,255,255),'2':(0,255,0),'3':(0,0,255),'4':(255,0,0)}

tilingParameters = json.load(open('datasets/tilingParameters.json'))
classifiedAndColoredMap = np.zeros((tilingParameters['height'], tilingParameters['width'],3))

for i in range(384):
    coords = tilingParameters['coordinates'][str(i)]
    classifiedAndColoredMap[coords['yLow']:coords['yHigh'], coords['xLow']:coords['xHigh']] += np.ones((512,512,3))*colorDict[f'{classificationfile[str(i)]}']

plt.matshow(classifiedAndColoredMap[0:7442,0:11138])
plt.show()