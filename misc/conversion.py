
from glob import glob
import numpy as np
from PIL import Image

'''pathToDatasets = 'datasets/cities'

pathToAllCities = glob(pathToDatasets+'/*')
for pathToCity in pathToAllCities:
    print(f"processing {pathToCity}")
    allTiles = glob(pathToCity+f'/*/*/*.tif')
    for tile in allTiles:
        pilImage = Image.open(tile,mode='r')
        print(f'Saving as {tile.split(".")[0]}.npy')
        npImage = np.asarray(pilImage, dtype=np.uint16)
        print(f'Saving as {tile.split(".")[0]}.npy')
        np.save(f'{tile.split(".")[0]}.npy', npImage)
        loadedImage = np.load('test.npy', allow_pickle=True)
        assert(npImage ==  loadedImage)'''

a = Image.open(r'C:\Users\hx21262\MAPHIS\datasets\cities\Luton\0105033050201.jpg')
a = np.asarray(a)
import matplotlib.pyplot as plt

plt.imshow(a)
plt.show()