#!/usr/bin/env python

import pandas as pd
from typing import List
import numpy as np
from typing import List

def loadCsvFile(filePath : str) -> pd.DataFrame:
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