import shutil
import os
import splitfolders

sourceNormal = "/Users/ramvegiraju/Desktop/personalProjects/COVIDX-RayDetection/dataset/normal"
sourceCovid = "/Users/ramvegiraju/Desktop/personalProjects/COVIDX-RayDetection/dataset/covid"
destNormalTrain = "/Users/ramvegiraju/Desktop/personalProjects/COVIDX-RayDetection/dataset/train/normal"
destCovidTrain = "/Users/ramvegiraju/Desktop/personalProjects/COVIDX-RayDetection/dataset/train/covid"

#print(os.listdir(sourceNormal))
#dest = shutil.move(sourceNormal, destNormalTrain)
print(os.listdir(sourceCovid))
dest = shutil.move(sourceCovid, destCovidTrain)