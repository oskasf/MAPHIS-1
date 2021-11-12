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

tolerance = 10

class Application(ttk.Frame):
    def __init__(self, master, rawFolderName, cityName, classifiedFolderPath, rawImagesFileExtension='npz', featureFileName='rawFeatures.csv'):
        ttk.Frame.__init__(self, master)

        self.master = master
        
        self.cityName = cityName
        self.rawFolderPath = Path(PurePath(rawFolderName).joinpath(cityName))
        self.classifiedFolderPath = Path(PurePath(classifiedFolderPath).joinpath(cityName))
        self.featureNamesListPath = classifiedFolderPath+'/featureListLabels.csv'
        
        Path(PurePath(classifiedFolderPath).joinpath(PurePath(cityName))).mkdir(parents=True, exist_ok=True)

        self.rawImagesFileExtension = rawImagesFileExtension
        self.featureFileName = featureFileName

        self.storedImages = glob.glob(str(self.rawFolderPath)+'/*.'+rawImagesFileExtension)
        self.nPatches = len(self.storedImages)
        
        if Path(PurePath(self.classifiedFolderPath).joinpath('lastIndex.csv')).is_file():
            with open(f'{self.classifiedFolderPath}lastIndex.csv', 'r') as csvFile:
                reader = csv.reader(csvFile)
                self.imageIndex = list(reader)[0]

        else:  
            self.imageIndex = 0

        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.fig.add_subplot(111).matshow(self.fileOpenFunction(self.storedImages[self.imageIndex]))
        
        self.canvas = FigureCanvasTkAgg(self.fig, master)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0,column=1)

        # Deal with buttons : put it on the grid (0,0) of the master by creating a Frame at that location
        rowButtonActions = 0
        rowIndexInfo = 1
        rowButtonPredefined0 = 2
        rowButtonPredefined1 = 3
        rowButtonPredefined2 = 4
        rowButtonPredefined3 = 5
        rowButtonPredefined4 = 6
        rowButtonPredefined5 = 7
        rowButtonPredefined6 = 8
        rowButtonSave = rowButtonPredefined6+ 1
        self.buttonFrame = tk.Frame(master)
        self.buttonFrame.grid(row=rowButtonActions,column=0)

        self.textField = ttk.Entry(self.buttonFrame , text="Input feature name")
        self.textField.grid(row=rowButtonActions,column=0)

        self.saveButton = tk.Button(self.buttonFrame, text="Save and Update", command=lambda:[self.save(self.textField.get(), self.storedProperties[self.imageIndex]), self.clearTextInput(self.textField), self.updateCanvas()])
        self.saveButton.grid(row=rowButtonActions,column=1)

        self.fpButton = tk.Button(self.buttonFrame, text="False Positive", command=lambda:[self.updateCanvas()])
        self.fpButton.grid(row=rowButtonActions,column=2)

        self.add_feature_button = tk.Button(self.buttonFrame, text="Add feature", command=lambda:[self.addFeatureToList(self.textField.get()), self.save(self.textField.get(), self.storedProperties[self.imageIndex]), self.clearTextInput(self.textField), self.updateCanvas()])
        self.add_feature_button.grid(row=rowButtonActions,column=3)

        self.currentIndexDisplay = tk.Label(self.buttonFrame, height = 1 , width = len(f'{self.nPatches}/ {self.nPatches}'), text=f'{self.imageIndex} / {self.nPatches}')
        self.currentIndexDisplay.grid(row=rowIndexInfo, column=0)

        self.indexJumpTextBox = ttk.Entry(self.buttonFrame , text="Go to index")
        self.indexJumpTextBox.grid(row=rowIndexInfo,column=1)

        self.indexJumpButton = ttk.Button(self.buttonFrame , text="Jump to index", command=lambda:[self.updateCanvas(int(self.indexJumpTextBox.get())), self.clearTextInput(self.indexJumpTextBox)])
        self.indexJumpButton.grid(row=rowIndexInfo,column=2)

        self.indexJumpButton = ttk.Button(self.buttonFrame , text="One before", command=lambda:[self.updateCanvas(self.imageIndex-1)])
        self.indexJumpButton.grid(row=rowIndexInfo,column=3)

        self.buttonLampPost = tk.Button(self.buttonFrame, text="L.P", command=lambda:[self.save("lamppost", self.storedProperties[self.imageIndex]), self.updateCanvas()])
        self.buttonLampPost.grid(row=rowButtonPredefined0,column=0)
        self.buttonManHole = tk.Button(self.buttonFrame, text="M.H", command=lambda:[self.save("manhole", self.storedProperties[self.imageIndex]), self.updateCanvas()])
        self.buttonManHole.grid(row=rowButtonPredefined0,column=1)
        self.buttonStone = tk.Button(self.buttonFrame, text="stone", command=lambda:[self.save("stone", self.storedProperties[self.imageIndex]), self.updateCanvas()])
        self.buttonStone.grid(row=rowButtonPredefined0,column=2)
        self.buttonChimney = tk.Button(self.buttonFrame, text="chimney", command=lambda:[self.save("chimney", self.storedProperties[self.imageIndex]),  self.updateCanvas()])
        self.buttonChimney.grid(row=rowButtonPredefined0,column=3)
        self.buttonChy = tk.Button(self.buttonFrame, text="chy.", command=lambda:[self.save("chy", self.storedProperties[self.imageIndex]),  self.updateCanvas()])
        self.buttonChy.grid(row=rowButtonPredefined1,column=0)
        self.buttonHotel = tk.Button(self.buttonFrame, text="hotel", command=lambda:[self.save("hotel", self.storedProperties[self.imageIndex]),  self.updateCanvas()])
        self.buttonHotel.grid(row=rowButtonPredefined1,column=1)
        self.buttonChurch = tk.Button(self.buttonFrame, text="church", command=lambda:[self.save("church", self.storedProperties[self.imageIndex]),  self.updateCanvas()])
        self.buttonChurch.grid(row=rowButtonPredefined1,column=2)
        self.buttonWorkshop = tk.Button(self.buttonFrame, text="workshop", command=lambda:[self.save("workshop", self.storedProperties[self.imageIndex]),  self.updateCanvas()])
        self.buttonWorkshop.grid(row=rowButtonPredefined1,column=3)
        self.buttonFirepost = tk.Button(self.buttonFrame, text="firepost", command=lambda:[self.save("firepost", self.storedProperties[self.imageIndex]),  self.updateCanvas()])
        self.buttonFirepost.grid(row=rowButtonPredefined2,column=0)
        self.buttonRiver = tk.Button(self.buttonFrame, text="river", command=lambda:[self.save("river", self.storedProperties[self.imageIndex]),  self.updateCanvas()])
        self.buttonRiver.grid(row=rowButtonPredefined2,column=1)
        self.buttonSchool = tk.Button(self.buttonFrame, text="school", command=lambda:[self.save("school", self.storedProperties[self.imageIndex]),  self.updateCanvas()])
        self.buttonSchool.grid(row=rowButtonPredefined2,column=2)
        self.buttonBarrack = tk.Button(self.buttonFrame, text="barrack", command=lambda:[self.save("barrack", self.storedProperties[self.imageIndex]),  self.updateCanvas()])
        self.buttonBarrack.grid(row=rowButtonPredefined2,column=3)
        self.buttonWorkhouse = tk.Button(self.buttonFrame, text="workhouse", command=lambda:[self.save("workhouse", self.storedProperties[self.imageIndex]),  self.updateCanvas()])
        self.buttonWorkhouse.grid(row=rowButtonPredefined3,column=0)
        self.buttonMarket = tk.Button(self.buttonFrame, text="market", command=lambda:[self.save("market", self.storedProperties[self.imageIndex]),  self.updateCanvas()])
        self.buttonMarket.grid(row=rowButtonPredefined3,column=1)
        self.buttonChapel = tk.Button(self.buttonFrame, text="chapel", command=lambda:[self.save("chapel", self.storedProperties[self.imageIndex]),  self.updateCanvas()])
        self.buttonChapel.grid(row=rowButtonPredefined3,column=2)
        self.buttonBank = tk.Button(self.buttonFrame, text="bank", command=lambda:[self.save("bank", self.storedProperties[self.imageIndex]),  self.updateCanvas()])
        self.buttonBank.grid(row=rowButtonPredefined3,column=3)
        self.buttonPub = tk.Button(self.buttonFrame, text="pub", command=lambda:[self.save("pub", self.storedProperties[self.imageIndex]),  self.updateCanvas()])
        self.buttonPub.grid(row=rowButtonPredefined4,column=0)
        self.buttonPublicHouse = tk.Button(self.buttonFrame, text="P.H", command=lambda:[self.save("publichouse", self.storedProperties[self.imageIndex]),  self.updateCanvas()])
        self.buttonPublicHouse.grid(row=rowButtonPredefined4,column=1)
        self.buttonHotel = tk.Button(self.buttonFrame, text="hotel", command=lambda:[self.save("hotel", self.storedProperties[self.imageIndex]),  self.updateCanvas()])
        self.buttonHotel.grid(row=rowButtonPredefined4,column=2)
        self.buttonInn = tk.Button(self.buttonFrame, text="inn", command=lambda:[self.save("inn", self.storedProperties[self.imageIndex]),  self.updateCanvas()])
        self.buttonInn.grid(row=rowButtonPredefined4,column=3)
        self.buttonBath = tk.Button(self.buttonFrame, text="bath", command=lambda:[self.save("bath", self.storedProperties[self.imageIndex]),  self.updateCanvas()])
        self.buttonBath.grid(row=rowButtonPredefined5,column=0)
        self.buttonTheatre = tk.Button(self.buttonFrame, text="theatre", command=lambda:[self.save("theatre", self.storedProperties[self.imageIndex]),  self.updateCanvas()])
        self.buttonTheatre.grid(row=rowButtonPredefined5,column=1)
        self.buttonPolice = tk.Button(self.buttonFrame, text="police", command=lambda:[self.save("police", self.storedProperties[self.imageIndex]),  self.updateCanvas()])
        self.buttonPolice.grid(row=rowButtonPredefined5,column=2)
        self.buttonWharf = tk.Button(self.buttonFrame, text="wharf", command=lambda:[self.save("wharf", self.storedProperties[self.imageIndex]),  self.updateCanvas()])
        self.buttonWharf.grid(row=rowButtonPredefined5,column=3)
        self.buttonYard = tk.Button(self.buttonFrame, text="yard", command=lambda:[self.save("yard", self.storedProperties[self.imageIndex]),  self.updateCanvas()])
        self.buttonYard.grid(row=rowButtonPredefined6,column=0)
        self.buttonGreen = tk.Button(self.buttonFrame, text="green", command=lambda:[self.save("green", self.storedProperties[self.imageIndex]),  self.updateCanvas()])
        self.buttonGreen.grid(row=rowButtonPredefined6,column=1)
        self.buttonPark = tk.Button(self.buttonFrame, text="park", command=lambda:[self.save("park", self.storedProperties[self.imageIndex]),  self.updateCanvas()])
        self.buttonPark.grid(row=rowButtonPredefined6,column=2)
        self.buttonQuarry = tk.Button(self.buttonFrame, text="quarry", command=lambda:[self.save("quarry", self.storedProperties[self.imageIndex]),  self.updateCanvas()])
        self.buttonQuarry.grid(row=rowButtonPredefined6,column=3)

        self.buttonSave = tk.Button(self.buttonFrame, text="Save Progression", command=lambda:[self.save("quarry", self.storedProperties[self.imageIndex]),  self.updateCanvas()])
        self.buttonSave.grid(row=rowButtonSave,column=3)

    def clearTextInput(self, textBoxAttribute):
        textBoxAttribute.delete(0, len(textBoxAttribute.get()))

    def fileOpenFunction(self, filePath):
        if self.rawImagesFileExtension == 'png':
            return Image.open(filePath)
        elif self.rawImagesFileExtension == 'npz':
            return np.load(filePath)['arr_0']
        else:
            print('Not Implemented')

    def updateCanvas(self, index =None):
        if index is not None:
            self.imageIndex =index
        else:
            self.imageIndex +=1
        if self.imageIndex == self.nPatches-1:
            self.master.destroy()

        self.fig.clear()
        self.fig.add_subplot(111).matshow(self.fileOpenFunction(self.storedImages[self.imageIndex]))
        self.canvas.draw()

        self.indexJumpTextBox.delete(0,len(self.indexJumpTextBox.get()))
        self.currentIndexDisplay['text'] = f'{self.imageIndex} / {self.nPatches}'

    def save(self, text, properties):
        fname = f'{self.classifiedFolderPath}/{text}s_{self.cityName}.csv'
        if not Path(fname).is_file():
            with open(fname, 'w',  newline='') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(self.header)

        with open(fname, 'r',  newline='') as csvFile:
            reader = csv.reader(csvFile)
            detectedFeaturesList = list(reader)

        if len(detectedFeaturesList)==1:
            with open(fname, 'a',  newline='') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(properties)
    
        else:
     
            for feature in detectedFeaturesList[1:]:
                print(feature)
                if int(feature[6])-tolerance<=int(properties[6])<=int(feature[6])+tolerance and int(feature[7])-tolerance<=int(properties[7])<=int(feature[7])+tolerance:
                    print(f'Feature already detected !')
                    return
                    
            with open(fname, 'a',  newline='') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(properties)

        newName = f'{self.classifiedFolderPath}/{text}'
        nFeatures = len(glob.glob(newName+'*.'+self.rawImagesFileExtension))

        shutil.copy(self.storedImages[self.imageIndex], newName+str(nFeatures)+'.'+self.rawImagesFileExtension)
        print(f"Copied {self.storedImages[self.imageIndex]} as {newName+str(nFeatures)}")
    
    def addFeatureToList(self, text):
        with open(self.featureNamesListPath, 'r',  newline='') as csvFile:
            reader = csv.reader(csvFile)
            featuresList = list(reader)

        if [text] not in featuresList:
            print(f'New feature found : {text}')
            with open(self.featureNamesListPath, 'a',  newline='') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow([text])
        else:
            print(f'Feature saved but not added to feature list. Cause : already exists')

    def save(self):
        with open(f'{self.classifiedFolderPath}lastIndex.csv', 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow([str(self.imageIndex)])
