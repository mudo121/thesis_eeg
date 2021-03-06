#!/usr/bin/env python

import pandas as pd
from typing import List
import numpy as np
from typing import List, Dict
import yaml
import io
import os

def readFileCSV(filePath : str) -> pd.DataFrame:
    ''' Loads a csv file and returns a pandas data frame '''
    if filePath.endswith('.csv'):
        return pd.read_csv(filePath)
    else:
        raise Exception("Given filepath does not end with .csv! Given filepath: {}".format(filePath))

def convertDataFrameToNumpyArray(pd : pd.DataFrame, channelNames : List) -> np.ndarray:
    ''' Converts a data frame into a numpy array '''
    npArrayList = []
    for channel in channelNames:
        npArrayList.append(pd[channel].to_numpy())
        
    return np.array(npArrayList)

def createChannelList(numberOfChannels: int) -> List:
    ''' Create a list of channel names channel_1, channel_2, etc. and returns as a list'''
    channelList = []
    for i in range(1, numberOfChannels+1):
        channelList.append("channel_{}".format(i))
    return channelList

def loadConfigFile(deviceName : str) -> Dict:
    ''' Load a yaml config file with the given device name '''

    if deviceName == "openBci":
        configFilePath = "D:/Masterthesis/thesis_eeg/config/openBci.yaml"
    elif deviceName == "muse-lsl":
        configFilePath = "D:/Masterthesis/thesis_eeg/config/muse_lsl.yaml"
    elif deviceName == "neuroscan":
        configFilePath = "D:/Masterthesis/thesis_eeg/config/neuroscan.yaml"
    elif deviceName == "muse_lsl_with_open_vibe":
        configFilePath = "D:/Masterthesis/thesis_eeg/config/muse_lsl_with_open_vibe.yaml"
    else:
        configFilePath = ""
        raise Exception("There is no confiFile for the device '{}'".format(deviceName))
    
    print ("Loading the config file for {}".format(deviceName))

    if not os.path.isfile(configFilePath):
        raise Exception("The Configfile '' does not exists!".format(configFilePath))

    with open(configFilePath, 'r') as stream:
        try:
            yamlConfig = yaml.safe_load(stream)
            return yamlConfig
        except yaml.YAMLError as exc:
            print(exc)
            return None

def filterDfByColumns(df : pd.DataFrame, columnsToKeep : List):
    ''' Filter the given df by the columns and keep only them
    
    E.g. columnsToKeep is "channel_1_" then every column gets deleted except this column
    '''
    if df is None:
        return df
    
    if columnsToKeep is None:
        return df
    
    requiredChannelList = []
    for dfColumn in df.columns:
        for requiredChannel in columnsToKeep:
            if requiredChannel in dfColumn:
                requiredChannelList.append(dfColumn)
        
    return df.filter(requiredChannelList)

def saveFeatureListToFile(featureList : List, filepath : str):
    ''' A list to the given file path - Used for saving feautres in a nice way for further processing'''
    if type(featureList) is not list:
        raise Exception("The given feature list is not a list!")
    
    print("Saving a feature list to: '{}'".format(filepath))
    
    f = open(filepath, "w")
    for feature in featureList:
        line = "{}\n".format(feature)
        f.write(line)
    f.close()


def saveDictToFile(myDict, filepath : str):
    ''' Save a dictionary the given file path'''
    
    print("Saving dict to {}".format(filepath))
    f = open(filepath, "w")
    for key, value in myDict.items():
        line = "{v} {k}\n".format(v=value, k=key.upper())
        f.write(str(line))
    f.close()


def loadTargetLabelsTxt(filePath : str) -> Dict:
    ''' Load a target tables txt file into a dict'''
    targetDict = {}
    with io.open(filePath, "r", newline=None) as fd:
        for line in fd:
            line = line.split()
            targetDict[line[1]] = line[0]
    
    return targetDict
    
    
def loadFeaturesTxt(filePath : str) -> List:
    ''' Load a features txt file into a list'''
    featureList = []
    with io.open(filePath, "r", newline=None) as fd:
        for line in fd:
            line = line.replace("\n", "")
            featureList.append(line)
    return featureList


def addFeatureToList(featureList, featureListIndex, newFeatureName):
    ''' Add a feature to a list, with a given index'''
    if newFeatureName not in featureList:
        featureList[featureListIndex] = newFeatureName
    return featureList

def createEmptyNumpyArray(d1, d2=None, d3=None):
    ''' Function to create an empty numpy Array, with the given dimesion'''
    if d2 is None and d3 is None:
        numpyArray = np.zeros((d1))
    
    elif d2 is not None and d3 is None:
        numpyArray = np.zeros((d1, d2))
    
    elif d3 is not None and d3 is not None:
        numpyArray = np.zeros((d1, d2, d3))
    
    print("Created Numpy Array - Shape: {}".format(numpyArray.shape))
    return numpyArray


def convert_filename_to_timestamp(filename):
    try:
        date = filename[filename.find('[')+1 : filename.find(']')].split('-')[0].replace('.','-')
        time = filename[filename.find('[')+1 : filename.find(']')].split('-')[1].replace('.',':')

        return pd.Timestamp('{} {}'.format(date, time))
    except Exception as e:
        print("Could not convert '{}' to timestamp".format(filename))
        print(e)
        return None

def load_data_from_subject_dir(subject_dir:str, file_name_includes:str = "driving", index_col:str ="Time:256Hz") -> List[pd.DataFrame]:
    ''' Loads all .csv files into a list of pandas dateframes from a given directory of a subject
    
    If file_name_includes is 'driving' then it will look for files, which are containing this name in the filename, and tries to load it as a dataframe
    '''
    data = []

    filesToExclude = ['complete', 'driving_fatigue.csv', 'driving_complete.csv', 'reaction_game_complete.csv']
    print("Files to exclude: {}".format(filesToExclude))
    
    for root, dirs, files in os.walk(subject_dir):
        for file in files:
            if file_name_includes in file and file not in filesToExclude: # check if this part is in the filename e.g. 'driving' but NOT 'complete' that's how the finall csv. gets named
                try:
                    print("Found '{}'".format(file))
                    csvPath = os.path.join(subject_dir, file)
                    
                    df = pd.read_csv(csvPath, index_col=index_col)
                    df.index = pd.to_datetime(df.index, unit='s', origin=convert_filename_to_timestamp(file))
                    data.append(df)
                    
                except Exception as e:
                    print("Could not load '{}' as csv!".format(csvPath))
                    raise(e)
    
    print("Found {} dataset(s) containing '{}'".format(len(data), file_name_includes))
    return data
