#!/usr/bin/env python

import pandas as pd
from typing import List
import numpy as np
from typing import List, Dict
import yaml

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

    if deviceName == "openBci":
        configFilePath = "D:/Masterthesis/thesis_eeg/config/openBci.yaml"
    elif deviceName == "muse-lsl":
        configFilePath = "D:/Masterthesis/thesis_eeg/config/muse_lsl.yaml"
    elif deviceName == "neuroscan":
        configFilePath = "D:/Masterthesis/thesis_eeg/config/neuroscan.yaml"
    else:
        configFilePath = ""
        raise Exception("There is no confiFile for the device '{}'".format(deviceName))
        return None
    
    print ("Loading the config file for {}".format(deviceName))

    with open(configFilePath, 'r') as stream:
        try:
            yamlConfig = yaml.safe_load(stream)
            return yamlConfig
        except yaml.YAMLError as exc:
            print(exc)
            return None

