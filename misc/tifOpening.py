from PIL import Image
from numpy import asarray, uint8
from cv2 import imread as cv2Imread
from tifffile import imread as tifffileImread

filePath = 'datasets/cities/Luton/500/tp_1/0105033010241.tif'

print('~~~~~ Opening with CV2 ~~~~~')
cv2Img = cv2Imread(filePath)
print('~~~~~ Opening with Tifffile ~~~~~')
tifImg = tifffileImread(filePath)
print('~~~~~ Opening with PIL + Numpy ~~~~~')
pilImg = Image.open(filePath) 
print('pilFile opened')
npImg = asarray(pilImg, dtype=uint8)