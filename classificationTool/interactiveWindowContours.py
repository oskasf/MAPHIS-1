from argparse import Action
import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showinfo
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import csv
import glob
from PIL import Image
from matplotlib.figure import Figure
import shutil
import numpy as np
from pathlib import Path, PurePath
from shutil import copyfile
from ast import literal_eval as make_tuple
import cv2

tolerance = 10

class Application(ttk.Frame):
    def __init__(self, classifType:str, master:tk.Tk, rawFolderName:str, cityName:str ,classifiedFolderName:str, mapsFileExtension='tif', contoursFileExtension='npz'):
        ttk.Frame.__init__(self, master)

        self.master = master
        self.classifType = classifType
        
        self.cityName = cityName
        self.rawFolderPath = Path(PurePath(rawFolderName).joinpath(cityName))
        self.classifiedFolderPath = Path(PurePath(classifiedFolderName).joinpath(cityName))
        self.saveProgressFile = self.classifiedFolderPath / f'progress{classifType.capitalize()}.csv'
        self.featureNamesListPath = self.classifiedFolderPath / f'featureList{classifType.capitalize()}.csv'
        
        self.classifiedFolderPath.mkdir(parents=True, exist_ok=True)

        self.mapsFileExtension= mapsFileExtension

        self.storedMaps = glob.glob(str(self.rawFolderPath)+'/*.'+mapsFileExtension)
        self.storedContours = glob.glob(str(self.rawFolderPath)+'/*.'+contoursFileExtension)
        self.nPatches = len(self.storedContours)

        self.heightThumbnail, self.widthThumbnail = 300

        self.currentContourName = self.storedContours[0]
        self.currentMapName = self.currentContourName.split('.')[0]+'.'+mapsFileExtension

        classifiedTileName = self.currentContourName.split('\\')[-1].split('.')[0].split('_')[0]
        
        if self.saveProgressFile.is_file():
            with open(self.saveProgressFile,'r') as f:
                reader = csv.reader(f)
                myList = list(reader)
                self.indexContour = int(myList[0][0])
        else:
            self.indexContour = 0

        self.figThumbnail = Figure(figsize=(5, 4), dpi=100)
        self.figThumbnail.add_subplot(111).matshow(self.drawContourOnMap(self.currentContourName))

        self.canvaThumbnail = FigureCanvasTkAgg(self.figThumbnail, master)
        self.canvaThumbnail.draw()
        self.canvaThumbnail.get_tk_widget().grid(row=0,column=1)

        self.canvaDrawned =  np.zeros((11400,7590,3), dtype=np.uint8)
        self.background = self.fileOpenFunction(self.currentMapName)
        cv2.CvtColor(self.background, cv2.COLOR_GRAY2RGB)

        '''dataframe = pd.read_csv('results/rawShapes/Barrow_in_Furness/shapeFeatures.csv', usecols = [i for i in range(4,15)])
        dataframe.hist( color='steelblue', edgecolor='black', linewidth=1.0, xlabelsize=8, ylabelsize=8, grid=False)
        plt.show()
        f, ax = plt.subplots(figsize=(10, 6))
        corr = dataframe.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        hm = sns.heatmap(round(corr,2), mask=mask,annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
                        linewidths=.05)
        f.subplots_adjust(top=0.93)
        t= f.suptitle('Shape Attributes Correlation Heatmap', fontsize=14)
        plt.show()
        '''
        # Deal with buttons : put it on the grid (0,0) of the master by creating a Frame at that location

        rowIndexInfo = 0
        rowTileInfo = 1
        rowButtonPredefined0 = 2

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
        self.saveButton = ttk.Button(self.buttonFrame , text="Save progress", command=lambda:[self.saveProgress()])
        self.saveButton.grid(row=rowTileInfo,column=0)

        ## contours cleaning buttons
        self.interesting = tk.Button(self.buttonFrame, text="Interesting", command=lambda:[self.save(), self.updateIndex()])
        self.interesting.grid(row=rowButtonPredefined0,column=0)
        self.notInteresting = tk.Button(self.buttonFrame, text="Not interesting", command=lambda:[self.save(), self.updateIndex()])
        self.notInteresting.grid(row=rowButtonPredefined0,column=1)

    def clearTextInput(self, textBoxAttribute):
        textBoxAttribute.delete(0, len(textBoxAttribute.get()))

    def fileOpenFunction(self, filePath:str):
        extension = Path(filePath).suffix
        if extension == '.png' or extension == '.tif':
            return np.asarray(Image.open(filePath))
        elif extension == 'npz':
            return np.load(filePath)['contour'], np.load(filePath)['features']
        else:
            raise NotImplementedError (f'{extension} opening function is not implemented.')

    def drawContourOnMap(self, filePath):
        contour, featureDict = self.fileOpenFunction(filePath)
        extentX = (featureDict['xTile']-featureDict['W']/2, featureDict['xTile']+featureDict['W']/2)
        extentY = (featureDict['yTile']-featureDict['H']/2, featureDict['yTile']+featureDict['H']/2)
        xBefore, xAfter = min(self.widthThumbnail, extentX[0]), min(self.widthThumbnail, 11400-extentX[1])
        yBefore, yAfter = min(self.heightThumbnail, extentY[0]), min(self.heightThumbnail, 7590 - extentY[1])
        cv2.drawContours(self.canvaDrawned, [contour], -1, (0,0,255))
        return (self.background+self.canvaDrawned)[extentX[0]-xBefore:extentX[0]+xAfter, extentY[0]-yBefore:extentY[0]+yAfter]

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

    

