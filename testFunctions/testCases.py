# %%
import testFunctions
import glob
import os
import funcUtils
import argparse

currentPath = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--cityName', default='Barrow_in_Furness', type=str, help='Name of the city of interest')
parser.add_argument('-l','--featureNames', nargs='+', help='List of features', default=['chimneys'], required=False)
parser.add_argument('--datasetPath', default= 'datasets/', required = False)
args = parser.parse_args()

def main():
    for featureName in args.featureNames:
        
        ## Necessary to get the path when different subdir names
        datasetPath = glob.glob(args.datasetPath+args.cityName+'/*/*/')[0]      

        ## Check if there is already a .csv file in the path, performs conversion otherwise
        if os.path.isfile(datasetPath+featureName+'-'+args.cityName+'.csv'):
            print('CSV file already exists')
        else:    
            path = datasetPath
            files = [i for i in os.listdir(path) if os.path.isfile(os.path.join(path,i)) and featureName+'-'+args.cityName in i]
            
            funcUtils.convertToCsV(datasetPath, files[0])

        nameDetected = datasetPath+'detected_'+featureName+'-'+args.cityName+'.csv'
        nameTarget = datasetPath+featureName+'-'+args.cityName+'.csv'
        testFunctions.assertResults('chimneys', nameDetected, nameTarget)

if __name__=='__main__':
    main()
