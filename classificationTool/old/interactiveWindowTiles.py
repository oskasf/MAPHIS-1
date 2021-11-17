import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showinfo
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import csv
import glob
from PIL import Image
from matplotlib.figure import Figure
import numpy as np
from pathlib import Path, PurePath
import json

class Application(ttk.Frame):
    def __init__(self, master:tk.Tk, cityName:str, datasetPath:Path, classifiedFolderPath:Path):
        ttk.Frame.__init__(self, master)

        self.master = master
        self.cityName = cityName
        self.datasetPath = datasetPath
        self.cityPath = next(datasetPath.glob(f'cities/{cityName}/*/*'))
        self.classifiedFolderPath = classifiedFolderPath
        self.featureNamesListPath = classifiedFolderPath / 'featureListTiles.csv'
        self.tilingParameters = json.load(open(datasetPath / 'tilingParameters.json'))
        
        print(self.tilingParameters)

        self.nRows, self.nCols,  = 5, 8
        self.heightThumbnail, self.widthThumbnail = int(7592/self.nRows),int(7592/self.nCols)

        self.currentTileName = self.storedImages[0]

        self.ColorDict = {'rich residential neighborhood':(128,128,128), 'poor residential neighborhood':(255,255,255), 'industrial district':(255,0,0),
                          'peri-urban district':(0,255,0),  'farm and forest':(0,0,255)}

        classifiedTileName = self.currentTileName.split('\\')[-1].split('.')[0]

        self.scaleFactor = 6
        
        if Path(PurePath(self.classifiedFolderPath).joinpath(classifiedTileName+'.npz')).is_file():
            loadPath = self.classifiedFolderPath / classifiedTileName
            self.colorisedTile = np.load(loadPath.with_suffix('.npz'))['arr_0']
            with open(loadPath.with_suffix('.csv'),'r') as f:
                reader = csv.reader(f)
                myList = list(reader)
                self.iTile, self.jTile = int(myList[0][0]), int(myList[1][0])
        else:
            self.colorisedTile = np.zeros((self.nRows*self.scaleFactor,self.nCols*self.scaleFactor,3), dtype=np.uint8)
            self.iTile, self.jTile = 0,0

        self.figThumbnail = Figure(figsize=(5, 4), dpi=100)
        self.figThumbnail.add_subplot(111).matshow(self.fileOpenFunction(self.currentTileName)[self.iTile:self.iTile+self.heightThumbnail, self.jTile:self.jTile+self.widthThumbnail])

        self.canvaThumbnail = FigureCanvasTkAgg(self.figThumbnail, master)
        self.canvaThumbnail.draw()
        self.canvaThumbnail.get_tk_widget().grid(row=0,column=1)

        self.figColorisedTile = Figure(figsize=(5,4), dpi=100)
        self.figColorisedTile.add_subplot(111).imshow(self.colorisedTile)

        self.canvaTile = FigureCanvasTkAgg(self.figColorisedTile, master)
        self.canvaTile.draw()
        self.canvaTile.get_tk_widget().grid(row=0, column=2)

        # Deal with buttons : put it on the grid (0,0) of the master by creating a Frame at that location

        rowIndexInfo = 0
        rowTileInfo = 1
        rowButtonPredefined0 = 2
        rowButtonPredefined1 = 3
        rowButtonPredefined2 = 4

        self.buttonFrame = tk.Frame(master)
        self.buttonFrame.grid(row=0,column=0)

        ## indexes
        self.currentIndexDisplay = tk.Label(self.buttonFrame, height = 1 , width = len(f'({self.iTile},{self.jTile}) / ({14},{18})'), text=f'({self.iTile},{self.jTile}) / ({14},{18})')
        self.currentIndexDisplay.grid(row=rowIndexInfo, column=0)

        self.indexJumpTextBox = ttk.Entry(self.buttonFrame , text="Go to index")
        self.indexJumpTextBox.grid(row=rowIndexInfo,column=1)
        
        self.indexJumpButton = ttk.Button(self.buttonFrame , text="Jump to index", command=lambda:[self.updateCanvas(indexString = self.indexJumpTextBox.get()), self.clearTextInput(self.indexJumpTextBox), self.updateIndex()])
        self.indexJumpButton.grid(row=rowIndexInfo,column=2)
        
        ## dropdown

        self.tileNameDisplayed = tk.StringVar(self.buttonFrame)
        self.tileNameDisplayed.set(self.currentTileName) # default value

        self.dropdownMenu = tk.OptionMenu(self.buttonFrame, self.tileNameDisplayed, *self.storedImages)
        self.dropdownMenu.grid(row=rowTileInfo,column=0)

        self.changeTileButton = ttk.Button(self.buttonFrame , text="Change Tile", command=lambda:[self.changeTile(), self.updateIndex()])
        self.changeTileButton.grid(row=rowTileInfo,column=1)

        self.saveButton = ttk.Button(self.buttonFrame , text="Save progress", command=lambda:[self.saveProgress()])
        self.saveButton.grid(row=rowTileInfo,column=2)

        ## Land classification tiles
        self.richResidential = tk.Button(self.buttonFrame, text="rich residential neighborhood", command=lambda:[self.updateCanvas(color=self.ColorDict["rich residential neighborhood"]), self.updateIndex()])
        self.richResidential.grid(row=rowButtonPredefined0,column=0)
        self.poorResidential = tk.Button(self.buttonFrame, text="poor residential neighborhood", command=lambda:[self.updateCanvas(color=self.ColorDict["poor residential neighborhood"]), self.updateIndex()])
        self.poorResidential.grid(row=rowButtonPredefined0,column=1)
        self.industrialDistrict = tk.Button(self.buttonFrame, text="industrial district", command=lambda:[self.updateCanvas(color=self.ColorDict["industrial district"]), self.updateIndex()])
        self.industrialDistrict.grid(row=rowButtonPredefined1,column=0)
        self.periUrban = tk.Button(self.buttonFrame, text="peri-urban district", command=lambda:[self.updateCanvas(color=self.ColorDict["peri-urban district"]), self.updateIndex()])
        self.periUrban.grid(row=rowButtonPredefined1,column=1)
        self.farmAndForest = tk.Button(self.buttonFrame, text="farm and forest", command=lambda:[self.updateCanvas(color=self.ColorDict["farm and forest"]), self.updateIndex()])
        self.farmAndForest.grid(row=rowButtonPredefined2,column=2, columnspan=2)

    def clearTextInput(self, textBoxAttribute):
        textBoxAttribute.delete(0, len(textBoxAttribute.get()))

    def fileOpenFunction(self, filePath):
        if self.rawImagesFileExtension == 'png' or self.rawImagesFileExtension == 'tif':
            return np.asarray(Image.open(filePath))
        elif self.rawImagesFileExtension == 'npz':
            return np.load(filePath)['arr_0']
        else:
            print('Not Implemented')

    def updateCanvas(self, color = None, indexString=None):
        if color:
            self.colorisedTile[self.iTile*self.scaleFactor:self.iTile*self.scaleFactor+self.scaleFactor, self.jTile*self.scaleFactor:self.jTile*self.scaleFactor+self.scaleFactor,:] = color

            self.jTile +=1
            if self.jTile == self.nCols:
                if self.iTile== self.nRows:
                    print('Reached the end of the tile')
                else:
                    self.jTile = 0
                    self.iTile +=1
                
            self.figThumbnail.clear()
            self.figThumbnail.add_subplot(111).matshow(self.fileOpenFunction(self.currentTileName)[self.iTile*self.heightThumbnail:self.iTile*self.heightThumbnail+self.heightThumbnail, self.jTile*self.widthThumbnail:self.jTile*self.widthThumbnail+self.widthThumbnail])
            self.canvaThumbnail.draw()

            self.figColorisedTile.clear()
            self.figColorisedTile.add_subplot(111).imshow(self.colorisedTile)
            self.canvaTile.draw()

        else:
            self.iTile = int(indexString.split(',')[0])
            self.jTile = int(indexString.split(',')[-1])
            self.figThumbnail.clear()
            self.figThumbnail.add_subplot(111).matshow(self.fileOpenFunction(self.currentTileName)[self.iTile*self.heightThumbnail:self.iTile*self.heightThumbnail+self.heightThumbnail, self.jTile*self.widthThumbnail:self.jTile*self.widthThumbnail+self.widthThumbnail])
            self.canvaThumbnail.draw()

    def changeTile(self):
        self.currentTileName =self.tileNameDisplayed.get()
        displayName = self.currentTileName.split('\\')[-1]

        if Path(PurePath(self.classifiedFolderPath).joinpath(displayName.split('.')[0]+'.npz')).is_file():
            loadPath = self.classifiedFolderPath / displayName.split('.')[0]
            self.colorisedTile = np.load(loadPath.with_suffix('.npz'))['arr_0']
            with open(loadPath.with_suffix('.csv'),'r') as f:
                reader = csv.reader(f)
                myList = list(reader)
                self.iTile, self.jTile = int(myList[0][0]), int(myList[1][0])
        else:
            self.colorisedTile = np.zeros((self.nRows*self.scaleFactor,self.nCols*self.scaleFactor,3), dtype=np.uint8)
            self.iTile, self.jTile = 0,0

        self.currentTileName =self.tileNameDisplayed.get()
        displayName = self.currentTileName.split('\\')[-1]
        print(f'Updating tile : {displayName}')
        self.figThumbnail.clear()
        self.figThumbnail.add_subplot(111).imshow(self.fileOpenFunction(self.currentTileName)[self.iTile:self.iTile+self.heightThumbnail, self.jTile:self.jTile+self.widthThumbnail])
        self.canvaThumbnail.draw()

        self.figColorisedTile.clear()
        self.figColorisedTile.add_subplot(111).imshow(self.colorisedTile)
        self.canvaTile.draw()

    def updateIndex(self):
        self.currentIndexDisplay['text'] = f'({self.iTile},{self.jTile}) / ({14},{18})'

    def saveProgress(self):
        displayName = self.currentTileName.split('\\')[-1]
        savePath = self.classifiedFolderPath / displayName.split('.')[0]
        np.savez(savePath, self.colorisedTile)
        with open(savePath.with_suffix('.csv'),'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([self.iTile])
            writer.writerow([self.jTile])

    

