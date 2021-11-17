import csv
import json
from PIL.TiffTags import TAGS
import zipfile
import glob
import os
from pathlib import Path, PurePath

def parseCSVtoJSON(csvPath, jsonPath):
    jsonDict = {}
    with open(csvPath, 'r') as csvFile:
        reader = csv.reader(csvFile)
        readerAsList = list(reader)
    header = readerAsList[0]
    for indexTown, town in enumerate(readerAsList[1:]):
        jsonDict[f'{indexTown}'] = {}
        for headerIndex, headerCategory in enumerate(header):
            jsonDict[f'{indexTown}'][headerCategory] = town[headerIndex]
    with open(jsonPath, 'w') as fp:
        json.dump(jsonDict, fp)

def getTiffProperties(tiffImage, showDict = False, returnDict=False):    
    meta_dict = {TAGS[key] : tiffImage.tag[key] for key in tiffImage.tag.keys()}
    if showDict:
        for key, value in meta_dict.items():
            print(' %s : %s' % (key, value))
    if returnDict:
        return meta_dict
    
def matchKeyToName(pathToJsonfile:str, key : str):
    cityKeysFile = json.load(open(pathToJsonfile))
    return cityKeysFile[key]['Town']

def createFolders(rawFolderPath='rawFolder', classifiedFolderPath='classifiedFolder', fileExtension='.zip'):

    listOfZips = glob.glob(rawFolderPath+'/*'+fileExtension)

    for zipName in listOfZips:
        ## Get dir name for folder creation in classified
        dirName = os.path.splitext(zipName)[0]
        Path(PurePath(classifiedFolderPath).joinpath(PurePath(dirName).name)).mkdir(parents=True, exist_ok=True)
        ## Extract compressed in raw

        with zipfile.ZipFile(zipName, 'r') as zip_ref:
            zip_ref.extractall(dirName)
        
'''def convertToCsV(datasetPath:str, fileName:str):
    """generates CSV file from other format

    Args:
        datasetPath (str): path to dataset
        fileName (str): name of the file to read from
    """
    fName = datasetPath+fileName
    if fileName.rsplit('.', 1)[1] == 'xlsx':
        iterator =pd.read_excel(fName).iterrows()
    
    elif fileName.rsplit('.', 1)[1] == 'txt':    
        iterator = pd.read_fwf(fName, index=False).iterrows()
    else:
        f'Wrong format'
    newHeader = ['x', 'y', 'city', 'tile', 'marker']
    with open(fName.rsplit('.', 1)[0]+'.csv','w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(newHeader)
        for _, row in iterator:
            r = row[0].split(',')
            newRow = [float(r[0]), float(r[1]), r[2], r[3][1:-1].split('.')[0], r[4]]
            writer.writerow(newRow)'''