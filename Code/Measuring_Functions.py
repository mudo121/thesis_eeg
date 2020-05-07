#!/usr/bin/env python

import pandas as pd
import statistics
from typing import List

'''
This file provides measuring functions
'''

def getChannelUsageInEpochSeries(epochSeries : pd.Series, featureSeries : bool, printUsage=False):
    ''' Count how often each channel is represented in the given sereies of epochs 
    
    @param epochSeries: A series of epochs, can be a normal epoch series or feature epoch series
    @param featureSereies: If true then we have to count different, 
                           because the feature epoch series has the channel in the index
                           (and the other in the columns)
    @parm printUsage: Prints the most used channels

    @return: A list of the most used channels by descending order
    ''' 

    foundChannels = {}

    if featureSeries: # feature epoch series
        for epoch in epochSeries: # loop through the epochs
            for index, row in epoch.iterrows(): # loop through the rows
                try:
                    foundChannels[index] += 1
                except KeyError: # if not in the dict, add it
                    foundChannels[index] = 1

    else: # normal epoch series
        for epoch in epochSeries:
            #for columns in epochSeries[1].columns:
            for columns in epoch.columns:
                try:
                    foundChannels[columns] += 1
                except KeyError: # if not in the dict, add it
                    foundChannels[columns] = 1


    #sortedFoundChannels = sorted(foundChannels.items(), key=lambda  item: item[1], reverse=True) # sort by found times

    mostUsedChannelsListDesc = []

    for key, value in sorted(foundChannels.items(), key=lambda  item: item[1], reverse=True): # sort by found times
        if printUsage:
            print("{} used {} times".format(key, value))

        mostUsedChannelsListDesc.append(key)

    return mostUsedChannelsListDesc


def faultyFeaturesNames(df : pd.DataFrame, maxPercentageMissing = 0.0) -> pd.Series:
    ''' Measure the most stable feature, by counting the NaN Values and return a series of columns which are not acceptable

    Return a list of column names where the maximum percentage is higher then the given parameter
    So the returned columns are not acceptable and have to be deleted.
    If no column name gets returned then every column is below or equal of 'maxPercentage'
    '''
    if df is None:
        return df
    
    percent_missing = df.isnull().sum() * 100 / len(df)
    missing_value_df = pd.DataFrame({'column_name': df.columns,
                                 'percent_missing': percent_missing})
    
    missing_value_df.sort_values('percent_missing', inplace=True)
    
    missing_value_df = missing_value_df[missing_value_df['percent_missing'] > maxPercentageMissing]
    return missing_value_df['column_name']

def countRecordsOfDf(df : pd.DataFrame) -> int:
    ''' Count a df with checks if its empty or None '''
    if df is None or df.empty:
        return 0
    else:
        return len(df)


def calculateMeanOverEpochs(valueList : List, numberOfEpochs : int = 5) -> List:
    ''' Calculate the mean over a given number of epochs
        e.g. if numberOfEpoch is 5 then the mean will get calculated from epoch 0-4; 5-9; 10-14; ...
        
    @param valueList: A list of values, where each entry represnts one epoch
    @param numberOfEpochs: A number which defines how many epochs should included in the calculations
    '''
    meanValueList = []
    start = 0
    end = numberOfEpochs
    for i in range(0, len(valueList), numberOfEpochs):
        meanValue = statistics.mean(valueList[start:end])
        meanValueList.append(meanValue)
        start += numberOfEpochs
        end += numberOfEpochs

    return meanValueList

def calculateStandardDeviationOverEpochs(valueList : List, numberOfEpochs : int = 5) -> List:
    ''' Calculate the standard deviation over a given number of epochs
        e.g. if numberOfEpoch is 5 then the standard deviation will get calculated from epoch 0-4; 5-9; 10-14; ...
        
    @param valueList: A list of values, where each entry represnts one epoch
    @param numberOfEpochs: A number which defines how many epochs should included in the calculations
    '''
    stDev_valueList = []
    start = 0
    end = numberOfEpochs
    for i in range(0, len(valueList), numberOfEpochs):
        value = statistics.stdev(valueList[start:end])
        stDev_valueList.append(value)
        start += numberOfEpochs
        end += numberOfEpochs

    return stDev_valueList