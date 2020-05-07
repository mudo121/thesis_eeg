#!/usr/bin/env python
'''
This file contains function which creates data, ready for machine learning
'''

import pandas as pd
import numpy as np
import os
from typing import List

# local imports
from pipelines import (filter_signal, pre_process_signal, feature_extraction)
from utils import readFileCSV, filterDfByColumns
from Measuring_Functions import faultyFeaturesNames, countRecordsOfDf
from consts import target_dict


def generateFeatureDf(csvFilePath, starttime, yamlConfig, label : str, generateData : bool = True, mainDir : str = None) -> (pd.Series, pd.DataFrame):
    ''' Generates dataframes of given csv filepaths
    These dataframes are containg features of the eeg data.

    It returns the series of epoch (of the eeg data) and the feature df
    '''
    
    if not generateData:
        # Load the data
        if mainDir is None:
            print("Enter a main directory where the pickle files can be found!")
            return None
        print ("Reading pickles files for {} data...".format(label))
        frequencyFeatureDf = pd.read_pickle(os.path.join(mainDir,'frequencyFeaturesDf_{}.pkl'.format(label)))
        epochSeries = pd.read_pickle(os.path.join(mainDir,'epochSeries_{}.pkl'.format(label)))
        return (epochSeries, frequencyFeatureDf)
        
    #######################
    #### GENERATE DATA ####
    # ---------------------
    print ("Generating {} driving feature df...".format(label))
    # load dataset
    df = readFileCSV(csvFilePath)  
    
    # Filter the signal
    df = filter_signal(df=df, config=yamlConfig, starttime=starttime)

    # Pre-process the signal
    epochSeries = pre_process_signal(df=df, config=yamlConfig)
    # Save epochSeries
    print("Saving epoch series...")
    epochSeries.to_pickle(os.path.join(mainDir,'epochSeries_{}.pkl'.format(label)))


    # Extract Frequency Features
    frequencyFeatureDf = feature_extraction(epochSeries=epochSeries, config=yamlConfig)
    # Save frequency features dataframe
    print("Saving frequency feature dataframe...")
    frequencyFeatureDf.to_pickle(os.path.join(mainDir,'frequencyFeaturesDf_{}.pkl'.format(label)))
    
    return (epochSeries, frequencyFeatureDf)


def createDataAndTargetArray(awakeDf, non_awakeDf, unlabeledDf, channelsToUse = None, splitChannels = False, maxPercentageNanValues = 0.0):
    '''  This functions creates a data and a target array which can be fed into classifiers
    
    The data array contains the data and the target array says what type of data it is e.g. awake or fatigue
    The index of the data and target array meatches each other. This is what this functions achives
    So the index 1 of the target array represents the index 1 at the data array. And so on. 

    So far NaN values are getting replaced by 0 in this process!
    
    @param awakeDf: A dataframe which only contains awake data/features
    @param non_awakeDf: A dataframe which only contains non awake data/features
    @param unlabeledDf: A dataframe which only contains unlabeled data/features
    @param channelsToUse: A list of which channels should be used | If none then use all channels
    @param splitChannels: If True then seperate the data and target by the channels we want to use | If false, it returns a list with one data array and one target array
    @param maxPercentageNanValues: Defines how many NaN Values are allowed to be NaN (in percentage!)
    
    @return ([dataArray], [targetArray])
    '''
    targetArray = []
    dataDf = pd.DataFrame()
    
    print("Creating Data and Target Array...")
    if channelsToUse is None:
        print("Using all channels...")
    else:
        usedChannelString = "Using the channels: "
        for channel in channelsToUse: usedChannelString += "{}, ".format(channel)
        print(usedChannelString)
            
    # Filter all dataframes - If the DF or channel is None, nothing will happen
    awakeDf = filterDfByColumns(awakeDf, channelsToUse)
    non_awakeDf = filterDfByColumns(non_awakeDf, channelsToUse)
    unlabeledDf = filterDfByColumns(unlabeledDf, channelsToUse)
    
    print ("Using Features where the NaN percentage is equal or lower than: {}".format(maxPercentageNanValues))
    
    
    if awakeDf is not None:
        
        # Filter the features
        awakeDf = awakeDf.drop(faultyFeaturesNames(awakeDf, maxPercentageMissing=maxPercentageNanValues), axis='columns')
        
        if not awakeDf.empty: # it is possible that the Df is now empty...
            # TODO - Checken ob gut oder schlecht?!?!?
            awakeDf = awakeDf.fillna(0)

            startCounter = 0
            if dataDf.empty:
                # Do this because then it will copy also all columns, then we can append stuff
                dataDf = awakeDf.loc[:1]

                # append to awake into the target array
                targetArray.append(target_dict['awake'])
                targetArray.append(target_dict['awake'])
                startCounter = 2

            # We have to start at 2 if we added 0:1 already
            for i in range(startCounter, len(awakeDf)):
                dataDf = dataDf.append(awakeDf.loc[i], ignore_index=True)
                targetArray.append(target_dict['awake'])

            print("Awake DF Columns: {}".format(awakeDf.columns))


            
    if non_awakeDf is not None:
        
        # Filter the features
        non_awakeDf = non_awakeDf.drop(faultyFeaturesNames(non_awakeDf, maxPercentageMissing=maxPercentageNanValues), axis='columns')
        if not non_awakeDf.empty: # it is possible that the Df is now empty...
        
            # TODO - Checken ob gut oder schlecht?!?!?
            non_awakeDf = non_awakeDf.fillna(0)
            
            startCounter = 0
            if dataDf.empty:
                # Do this because then it will copy also all columns, then we can append stuff
                dataDf = non_awakeDf.loc[:1]

                # append to awake into the target array
                targetArray.append(target_dict['non-awake'])
                targetArray.append(target_dict['non-awake'])
                startCounter = 2

            # We have to start at 2 if we added 0:1 already
            for i in range(startCounter, len(non_awakeDf)):
                dataDf = dataDf.append(non_awakeDf.loc[i], ignore_index=True)
                targetArray.append(target_dict['non-awake'])

            print("Non-Awake DF Columns: {}".format(non_awakeDf.columns))
            
            
    if unlabeledDf is not None:
        
        # Filter the features
        unlabeledDf = unlabeledDf.drop(faultyFeaturesNames(unlabeledDf, maxPercentageMissing=maxPercentageNanValues), axis='columns')
        
        if not unlabeledDf.empty: # it is possible that the Df is now empty...
            # TODO - Checken ob gut oder schlecht?!?!?
            unlabeledDf = unlabeledDf.fillna(0)

            startCounter = 0
            if dataDf.empty:
                # Do this because then it will copy also all columns, then we can append stuff
                dataDf = unlabeledDf.loc[:1]

                # append to awake into the target array
                targetArray.append(target_dict['unlabeled'])
                targetArray.append(target_dict['unlabeled'])
                startCounter = 2

            # We have to start at 2 if we added 0:1 already
            for i in range(startCounter, len(unlabeledDf)):
                dataDf = dataDf.append(unlabeledDf.loc[i], ignore_index=True)
                targetArray.append(target_dict['unlabeled'])

            print("Unlabled DF Columns: {}".format(unlabeledDf.columns))

    # Information about the Arrays
    awakeRecords = countRecordsOfDf(awakeDf)
    non_awakeRecords = countRecordsOfDf(non_awakeDf)
    unlabeledRecords = countRecordsOfDf(unlabeledDf)
    
    print("""The Data/Target Array contains:
    - {awake} awake data records
    - {non_awake} non-awake data records
    - {unlabeled} unlabeled data records""".format(awake = awakeRecords,
                                                   non_awake = non_awakeRecords,
                                                   unlabeled = unlabeledRecords))

    return (dataDf.to_numpy(), np.array(targetArray))




