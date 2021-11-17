from abc import ABC,ABCMeta, abstractmethod
import csv
import io
import pandas as pd
import time

class TEST(ABC):
    """Abstract class for the test

    Args:
        not sure if this works as it is supposed to but here we are
    """
    @property
    def name(self):
        pass

    @property
    def acceptanceMargin(self):
        pass

    @abstractmethod
    def test(self, featureName,  resultDetected, resultTarget):        
        f'--------- Beginning {self.name} Test ---------'

class isFeatureWellPositioned(TEST):
    def __init__(self):
        self._name = 'Feature well Positioned'
        self._acceptanceMargin = 0

    def test(self, featureName:str,  resultDetected:pd.DataFrame, resultTarget:pd.DataFrame, log:io.TextIOWrapper):
        """Implements test pass. This test checks if the features are well positioned in the tiles.

        Args:
            featureName (str): name of the feature 
            resultDetected (pd.DataFrame): data frame containing the results of the detection
            resultTarget (pd.DataFrame): data frame containing the target positions
            log (io.TextIOWrapper): log to write results in 
        """
        detectionResults = {'well positioned' : 0,
                            'badly positioned'  : 0}
        tilesThatRequireAttention = {'badly positioned'  : []}
        f'--------- Beginning {self._name} Test ---------'
        log.write(f'--------- Beginning {self._name} Test ---------\n')
        for index, row in resultDetected.iterrows():            
            x,y,h,w = row[0],row[1],row[4],row[5]
            xMin, xMax, yMin, yMax = x-w, x+2*w, y-h, y+2*h
            if resultTarget.query('@xMin<x<@xMax and @yMin<y<@yMax').empty:
                detectionResults['badly positioned'] += 1
                tilesThatRequireAttention['badly positioned'].append(row[3])
            else:
                detectionResults['well positioned'] += 1
        f'The full test results are : '
        log.write(f'The full test results are : \n')
        for key, value in detectionResults.items():
            f'There are {value} {key} {featureName}'
            log.write(f'There are {value} {key} {featureName}\n')
        if not tilesThatRequireAttention['badly positioned'] :
            f"No tiles require our attention"
            log.write(f"No tiles require our attention\n")
        else:
            for key, value in tilesThatRequireAttention.items():
                f'Tiles {key} : '
                log.write(f'Tiles {key} : \n')
                for tN in value:
                    log.write(f'{tN}\t')


class isFeatureInMap(TEST):
    def __init__(self):
        self._name = 'Feature In Tile'
        self._acceptanceMargin = 0

    def test(self, featureName:str,  resultDetected:pd.DataFrame, resultTarget:pd.DataFrame, log:io.TextIOWrapper):
        """Implements test pass. This test checks for each tile if there are the same amount of features in each tiles in the detected dataframe as there are in the target dataframe.

        Args:
            featureName (str): name of the feature 
            resultDetected (pd.DataFrame): data frame containing the results of the detection
            resultTarget (pd.DataFrame): data frame containing the target positions
            log (io.TextIOWrapper): log to write results in 
        """
        nFeaturesDetected = resultDetected.shape[0]
        nFeaturesTarget = resultTarget.shape[0]
        detectionResults = {'well detected' : 0,
                            'under detected'  : 0,
                            'over detected' : 0}
        tilesThatRequireAttention = {'under detected'  : [],
                                     'over detected'  : []}
        f'--------- Beginning {self._name} Test ---------'
        f"Number of different tiles with Target {featureName} : {resultTarget['tile'].nunique()}"
        f"Number of different tiles with detected {featureName} : {resultDetected['tile'].nunique()}"
        log.write(f'--------- Beginning {self._name} Test ---------\n')
        log.write(f"Number of different tiles with Target {featureName} : {resultTarget['tile'].nunique()}\n")
        log.write(f"Number of different tiles with detected {featureName} : {resultDetected['tile'].nunique()}\n")
        for tileNameTarget, tileValueTarget in resultTarget['tile'].value_counts().items():   
            if tileNameTarget in resultDetected['tile'].value_counts():
                tileValueDetected = resultDetected['tile'].value_counts()[tileNameTarget]
                if (tileValueDetected==tileValueTarget):
                    f"Test passed : there are {tileValueDetected} detected {featureName} in tile {tileNameTarget}"
                    log.write(f"Test passed : there are {tileValueDetected} detected {featureName} in tile {tileNameTarget}\n")
                    detectionResults['well detected'] += tileValueDetected
                elif (tileValueDetected<tileValueTarget):
                    f"Test Failed : there are {tileValueDetected} detected {featureName} in tile {tileNameTarget} but there should be {tileValueTarget}"
                    f"There are {tileValueTarget-tileValueDetected} undetected {featureName}  in that map"
                    log.write(f"Test Failed : there are {tileValueDetected} detected {featureName} in tile {tileNameTarget} but there should be {tileValueTarget}\n")
                    log.write(f"There are {tileValueTarget-tileValueDetected} undetected {featureName}  in that map\n")
                    detectionResults['under detected'] += tileValueTarget-tileValueDetected
                    tilesThatRequireAttention['under detected'].append(tileNameTarget)
                elif (tileValueDetected>tileValueTarget):
                    f"Test Failed : there are {tileValueDetected} detected {featureName} in tile {tileNameTarget} but there should be {tileValueTarget}"
                    f"There are {tileValueDetected-tileValueTarget} overdetected {featureName} in that map"
                    log.write(f"Test Failed : there are {tileValueDetected} detected {featureName} in tile {tileNameTarget} but there should be {tileValueTarget}\n")
                    log.write(f"There are {tileValueDetected-tileValueTarget} overdetected {featureName} in that map\n")
                    detectionResults['over detected'] += tileValueDetected-tileValueTarget
                    tilesThatRequireAttention['over detected'].append(tileNameTarget)
            else:
                f"Test Failed : No detected {featureName} in tile {tileNameTarget}"
                log.write(f"Test Failed : No detected {featureName} in tile {tileNameTarget}\n")
        f'The full test results are : '
        f'Number of detected {featureName} : {nFeaturesDetected}'
        f'Target number of {featureName} : {nFeaturesTarget}'
        log.write(f'The full test results are : \n')
        log.write(f'Number of detected {featureName} : {nFeaturesDetected}\n')
        log.write(f'Target number of {featureName} : {nFeaturesTarget}\n')
        for key, value in detectionResults.items():
            f'There are {value} {key} {featureName}'
            log.write(f'There are {value} {key} {featureName}\n')
        if nFeaturesTarget==nFeaturesDetected:
            f"No tiles require our attention"
            log.write(f"No tiles require our attention\n")
        else:
            for key, value in tilesThatRequireAttention.items():
                f'Tiles {key} : '
                log.write(f'Tiles {key} : \n')
                tiles = ''
                for tN in value:
                    tiles+=tN+' '
                f'{tiles}'
                log.write(f'{tiles}\n')
        
class numberOfFeatures(TEST):
    def __init__(self):
        self._name = 'Number of Features'
        self._acceptanceMargin = 0.1
        
    def test(self, featureName:str, resultDetected:pd.DataFrame, resultTarget:pd.DataFrame, log:io.TextIOWrapper):
        """Implements test pass. This test checks for each tile if there are the same amount of features in the detected dataframe as there are in the target dataframe.

        Args:
            featureName (str): name of the feature 
            resultDetected (pd.DataFrame): data frame containing the results of the detection
            resultTarget (pd.DataFrame): data frame containing the target positions
            log (io.TextIOWrapper): log to write results in 
        """
        try:
            nFeaturesDetected = resultDetected.shape[0]
            nFeaturesTarget = resultTarget.shape[0]
            f'--------- Beginning {self._name} Test ---------'
            f'Target number of {featureName} : {nFeaturesTarget}'
            f'Number of detected {featureName} :  {nFeaturesDetected}'
            log.write(f'--------- Beginning {self._name} Test ---------\n')
            log.write(f'Target number of {featureName} : {nFeaturesTarget}\n')
            log.write(f'Number of detected {featureName} :  {nFeaturesDetected}\n')
            accuracyRatio = nFeaturesTarget / nFeaturesDetected
            if 1-self._acceptanceMargin <= accuracyRatio <= 1+self._acceptanceMargin:
                f'{self._name} test passed with {accuracyRatio*100} accuracy.'
                log.write(f'{self._name} test passed with {accuracyRatio*100} accuracy.\n')
                if 1-self._acceptanceMargin < accuracyRatio < 1.0:
                    f"However, there are {nFeaturesDetected - nFeaturesTarget} {featureName} not detected."
                    log.write(f"However, there are {nFeaturesDetected - nFeaturesTarget} {featureName} not detected.\n")
                elif 1 < accuracyRatio < 1.0+self._acceptanceMargin :
                    f"However, there are {nFeaturesTarget - nFeaturesDetected } {featureName} over detected."
                    log.write(f"However, there are {nFeaturesTarget - nFeaturesDetected } {featureName} over detected.\n")
            else:
                f'{self._name} test failed. '
                log.write(f'{self._name} test failed.\n')
                if accuracyRatio < 1.0: 
                    f"There are {nFeaturesDetected - nFeaturesTarget} detected {featureName}."
                    log.write(f"There are {nFeaturesDetected - nFeaturesTarget} detected {featureName}.\n")
                else:
                    f"There are {nFeaturesTarget - nFeaturesDetected }undetected {featureName}."
                    log.write(f"There are {nFeaturesTarget - nFeaturesDetected } undetected {featureName}.\n")
        except nFeaturesDetected == 0:
            f"There are no detected Features"
            log.write(f"There are no detected Features\n")
        except nFeaturesTarget == 0:
            f"There are no target Features"
            log.write(f"There are no target Features\n")
        
def writeDetectionResultsDict(detectionDict:dict, filePath:str, cityName:str):
    """Writes the detection results in one csv file for each DETECTED feature name

    Args:
        detectionDict (dict): dict {featureName : featuresList}
        filePath (str): path to write results to
        cityName (str): name of the city of interest
    """
    fp = filePath
    header = ['x', 'y', 'city', 'tile', 'boxH', 'boxW', 'confidence Score', 'xTile', 'yTile']

    for featureName, featureList in detectionDict.items():
        fName = fp + 'detected_'+featureName+'s-' +cityName +'.csv'

        with open(fName,'w', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(header)

            for detectedElement in featureList:
                writer.writerow(detectedElement)

def checkIfAlreadyDetected(detectedFeaturesList:list, x:float, y:float, cityName:str, tileName:str, H:int, W:int, confidenceScore:float, xTile:int, yTile:int,  featureName:str) -> list:
    """Checks if a feature has already been detected (necessary as thumbnails overlap)

    Args:
        detectedFeaturesList (list): list of detected features 
        x (float): x position in the MAP
        y (float): y position in the MAP
        cityName (str): name of the city 
        tileName (str): name of the map
        H (int): height of the bounding box
        W (int): width of the bounding box
        confidenceScore (float): classification confidence score
        xTile (int): x position in the TILE
        yTile (int): y position in the TILE
        featureName (str): name of the feature

    Returns:
        list : list of detected features, same as detectedFeaturesList if the feature is already in the list.
    """
    tolerance = 5
    if bool(detectedFeaturesList)==False:
        detectedFeaturesList.append([x,y,'\''+cityName+'\'',tileName,H,W, confidenceScore, xTile, yTile])
        f'Feature {featureName} added because of empty list with confidence {confidenceScore}' 
        return detectedFeaturesList
    else:
        for feature in detectedFeaturesList:
            if feature[0]-tolerance<=x<=feature[0]+tolerance and feature[1]-tolerance<=y<=feature[1]+tolerance:
                f'Feature already detected !'
                return detectedFeaturesList
        detectedFeaturesList.append([x,y,'\''+cityName+'\'',tileName,H,W, confidenceScore, xTile, yTile])    
        f'Feature {featureName}  added with confidence {confidenceScore}' 
        return detectedFeaturesList

def assertResults(featureName:str, detectedFeaturesFile:pd.DataFrame, targetFeaturesFile: pd.DataFrame):
    """performs test results assertion once tests have been performed

    Args:
        featureName (str): name of the feature of interest
        detectedFeaturesFile (pd.DataFrame): data frame containing the results of the detection
        targetFeaturesFile (pd.DataFrame): data frame containing the target positions
    """
    log = open(f'./results/test_results_'+featureName+'.txt', 'a')
    dashed_line = '-' * 80
    head = f'Begginning tests for feature {featureName} at time {time.asctime(time.gmtime(time.time()))}'
    print(f'{dashed_line}\n{head}\n{dashed_line}')
    log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

    numberOfFeaturesTest = numberOfFeatures()
    isFeatureInMapTest = isFeatureInMap()
    isFeatureWellPositionedTest = isFeatureWellPositioned()

    resultDetected = pd.read_csv(detectedFeaturesFile)    
    resultTarget = pd.read_csv(targetFeaturesFile)

    numberOfFeaturesTest.test(featureName, resultDetected, resultTarget, log)
    isFeatureInMapTest.test(featureName, resultDetected, resultTarget, log)
    isFeatureWellPositionedTest.test(featureName, resultDetected, resultTarget, log)
        
    log.close()
        
