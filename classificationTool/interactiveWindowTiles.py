import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showinfo
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import csv
import glob
from PIL import Image
from matplotlib.figure import Figure
import numpy as np
from pathlib import Path
import json

class Application(ttk.Frame):
    def __init__(self, master:tk.Tk, cityName:str, datasetPath:Path, tileFileFormat:str):
        super(Application, self).__init__()

        self.cityName = cityName
        self.datasetPath = datasetPath
        self.cityPath = next(datasetPath.glob(f'cities/{cityName}/*/*'))

        self.allTilesPath = list(self.cityPath.glob(f'*{tileFileFormat}'))
        self.allTilesNames = [cityPath.stem for cityPath in self.allTilesPath]

        self.classifiedFolderPath = Path(f'{datasetPath}/classifiedMaps/{cityName}')
        self.classifiedFolderPath.mkdir(parents=True, exist_ok=True)
        self.tilingParameters = json.load(open(datasetPath / 'tilingParameters.json'))
        self.nTiles = int(self.tilingParameters["nCols"]*self.tilingParameters["nRows"])
        
        self.setCurrentlyOpenedFile(self.allTilesNames[0])
        self.loadClassifiedDict()

        self.classes = {'rich residential neighborhood':0, 'poor residential neighborhood':1, 'industrial district':2,
                          'peri-urban district':3,  'farm and forest':4}

        self.classifiedTile = {}        

        self.figThumbnail = Figure(figsize=(5, 4), dpi=100)
        self.canvaThumbnail = FigureCanvasTkAgg(self.figThumbnail, master)
        self.canvaThumbnail.get_tk_widget().grid(row=0,column=1)
        self.displayThumbnail()

        # Deal with buttons : put it on the grid (0,0) of the master by creating a Frame at that location

        rowIndexInfo = 0
        rowTileInfo = 1
        rowButtonPredefined0 = 2
        rowButtonPredefined1 = 3
        rowButtonPredefined2 = 4

        self.buttonFrame = tk.Frame(master)
        self.buttonFrame.grid(row=0,column=0)

        ## indexes
        self.currentIndexDisplay = tk.Label(self.buttonFrame, height = 1 , width = len(f'({self.currentThumbnailIndex}) / ({self.nTiles})'), text=f'({self.currentThumbnailIndex}) / ({self.nTiles})')
        self.currentIndexDisplay.grid(row=rowIndexInfo, column=0)

        self.indexJumpTextBox = ttk.Entry(self.buttonFrame , text="Go to index")
        self.indexJumpTextBox.grid(row=rowIndexInfo,column=1)
        
        self.indexJumpButton = ttk.Button(self.buttonFrame , text="Jump to index", command=lambda:[self.updateCanvas(indexString = self.indexJumpTextBox.get()), self.clearTextInput(self.indexJumpTextBox), self.updateIndex()])
        self.indexJumpButton.grid(row=rowIndexInfo,column=2)
        
        ## dropdown

        self.tileNameDisplayed = tk.StringVar(self.buttonFrame)
        self.tileNameDisplayed.set(self.currentTileName) # default value

        self.dropdownMenu = tk.OptionMenu(self.buttonFrame, self.tileNameDisplayed, *self.allTilesNames)
        self.dropdownMenu.grid(row=rowTileInfo,column=0)

        self.changeTileButton = ttk.Button(self.buttonFrame , text="Change Tile", command=lambda:[self.changeTile(), self.updateIndex()])
        self.changeTileButton.grid(row=rowTileInfo,column=1)

        self.saveButton = ttk.Button(self.buttonFrame , text="Save progress", command=lambda:[self.saveProgress()])
        self.saveButton.grid(row=rowTileInfo,column=2)

        rowButtonPredefined0 = 2
        rowButtonPredefined1 = 3
        rowButtonPredefined2 = 4

        ## Land classification tiles
        self.richResidential = tk.Button(self.buttonFrame, text="rich residential neighborhood", command=lambda:[self.classify(self.classes["rich residential neighborhood"]), self.updateCanvas(), self.updateIndex()])
        self.richResidential.grid(row=rowButtonPredefined0,column=0)
        self.poorResidential = tk.Button(self.buttonFrame, text="poor residential neighborhood", command=lambda:[self.classify(self.classes["poor residential neighborhood"]), self.updateCanvas(), self.updateIndex()])
        self.poorResidential.grid(row=rowButtonPredefined0,column=1)
        self.industrialDistrict = tk.Button(self.buttonFrame, text="industrial district", command=lambda:[self.classify(self.classes["industrial district"]), self.updateCanvas(), self.updateIndex()])
        self.industrialDistrict.grid(row=rowButtonPredefined1,column=0)
        self.periUrban = tk.Button(self.buttonFrame, text="peri-urban district", command=lambda:[self.classify(self.classes["peri-urban district"]), self.updateCanvas(), self.updateIndex()])
        self.periUrban.grid(row=rowButtonPredefined1,column=1)
        self.farmAndForest = tk.Button(self.buttonFrame, text="farm and forest", command=lambda:[self.classify(self.classes["farm and forest"]), self.updateCanvas(), self.updateIndex()])
        self.farmAndForest.grid(row=rowButtonPredefined2,column=2, columnspan=2)

    def clearTextInput(self, textBoxAttribute):
        textBoxAttribute.delete(0, len(textBoxAttribute.get()))

    def fileOpenFunction(self, filePath:Path) -> np.float32:
        if filePath.suffix in ['.png', '.tif', '.jpg']:
            array = np.asarray(Image.open(filePath), np.float32)
            paddedArray = np.pad(array, ((self.tilingParameters['paddingX'], self.tilingParameters['paddingX']),(self.tilingParameters['paddingY'], self.tilingParameters['paddingY'])), 'constant', constant_values=255)
            return paddedArray
        elif filePath.suffix == '.npz':
            return np.load(filePath, np.float32)['arr_0']
        else:
            print('Not Implemented')

    def classify(self, savedClass:int):
        self.classifiedDict[f'{self.currentThumbnailIndex}'] = savedClass
        try:
            self.classifiedDict['Not Classified'].remove(self.currentThumbnailIndex)
            self.classifiedDict['Classified'].append(self.currentThumbnailIndex)
        except ValueError:
            pass

    def updateCanvas(self, indexString=None):
        if indexString is not None:
            if indexString.isdigit() and int(indexString) < self.nTiles:
                    self.currentThumbnailIndex = int(indexString)
            else:
                print(f"Enter positive integers inferior to {self.nTiles} Only")
        else:
            if self.currentThumbnailIndex + 1< self.nTiles:
                self.currentThumbnailIndex+= 1
            else:
                print(f"Reached the end of the tile, not updating index")

        self.displayThumbnail()
    
    def changeTile(self):
        self.setCurrentlyOpenedFile(self.tileNameDisplayed.get())
        self.loadClassifiedDict()
        self.displayThumbnail()

    def updateIndex(self):
        self.currentIndexDisplay['text'] = f'({self.currentThumbnailIndex}) / ({self.nTiles})'

    def saveProgress(self):
        with open(self.classifiedFolderPath / f'{self.currentTileName}.json', 'w') as outfile:
            json.dump(self.classifiedDict, outfile)
        print(f'Progress saved in {self.currentTileName}.json')

    def setCurrentlyOpenedFile(self, tileName:str):
        self.currentTileName = tileName
        self.currentIndex = self.allTilesNames.index(self.currentTileName)
        self.currentTilePath = self.allTilesPath[self.currentIndex]
        self.currentlyOpenedFile = self.fileOpenFunction(self.currentTilePath)

    def getCurrentCoordinates(self)->dict:
        return self.tilingParameters['coordinates'][f'{self.currentThumbnailIndex}']

    def displayThumbnail(self):
        self.currentCoordinates = self.getCurrentCoordinates()
        self.figThumbnail.clear()
        self.figThumbnail.add_subplot(111).imshow(self.currentlyOpenedFile[self.currentCoordinates['yLow']:self.currentCoordinates['yHigh'], self.currentCoordinates['xLow']:self.currentCoordinates['xHigh']])
        self.canvaThumbnail.draw()

    def loadClassifiedDict(self):
        if Path(self.classifiedFolderPath / f'{self.currentTileName}.json').is_file()==False:
            self.classifiedDict = {'Not Classified':[i for i in range(self.nTiles)],
                                'Classified': []}
            with open(self.classifiedFolderPath / f'{self.currentTileName}.json', 'w') as outfile:
                json.dump(self.classifiedDict, outfile)
        else:
            self.classifiedDict = json.load(open(self.classifiedFolderPath / f'{self.currentTileName}.json'))

        if len(self.classifiedDict['Not Classified']) !=0:
            self.currentThumbnailIndex = self.classifiedDict['Not Classified'][0]
        else:
            self.currentThumbnailIndex = self.classifiedDict['Classified'][0]
