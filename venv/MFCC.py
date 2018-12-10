import numpy as np
import librosa
import tensorflow as tf
import os
import shutil

def mfcc(fileName):
    # all of the test records are sampled with rate 22050
    y, sr = librosa.load('sa1.wav')
    return librosa.feature.mfcc(y=y, sr=sr,n_mfcc=13)

def initialFiles(originalTimitDir):
    listOfFolders = os.listdir(originalTimitDir)
    print(len(listOfFolders))
    count = 0
    for folder in listOfFolders:
        newDir = "wavFiles/"+ str(count)
        os.mkdir(newDir)
        listOfFiles = os.listdir(originalTimitDir + "/" + folder)
        fileCount = 0
        for file in listOfFiles:
            if os.path.splitext(file)[1] == ".wav":
                shutil.copyfile(originalTimitDir + "/" + folder+ "/" + file, newDir+ "/"+ str(fileCount) + ".wav")
                fileCount += 1
        count += 1


# os.mkdir("wavFiles")
# initialFiles("F:\武汉大学\物联网\学习\\2016-2017\毕业设计\\timit_test_1-3\dr1")