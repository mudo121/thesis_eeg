#!/usr/bin/env python
'''
This file contains function which creates data, ready for machine learning
'''

import pandas as pd
import os

# local imports
from pipelines import (filter_signal, pre_process_signal, feature_extraction)
from utils import readFileCSV

def generateFeatureDf(csvFilePath, starttime, yamlConfig, label : str, generateData : bool = True, mainDir : str = None) -> pd.DataFrame:
    ''' Generates datframes of given csv filepaths'''
    
    if not generateData:
        if mainDir is None:
            print("Enter a main directory where the pickle files can be found!")
            return None
        print ("Reading pickles files for {} data...".format(label))
        frequencyFeatureDf = pd.read_pickle(os.path.join(mainDir,'generatedData','frequencyFeaturesDf_{}.pkl'.format(label)))
        epochSeries = pd.read_pickle(os.path.join(mainDir,'generatedData','epochSeries_{}.pkl'.format(label)))
        return (epochSeries, frequencyFeatureDf)
        
    # ### GENERATE DATA ###
    print ("Generating {} driving feature df...".format(label))
    # load dataset
    df = readFileCSV(csvFilePath)  
    
    # Filter the signal
    df = filter_signal(df=df, config=yamlConfig, starttime=starttime)

    # Pre-process the signal
    epochSeries = pre_process_signal(df=df, config=yamlConfig)
    # Save epochSeries
    print("Saving epoch series...")
    epochSeries.to_pickle(os.path.join(mainDir,'generatedData','epochSeries_{}.pkl'.format(label)))


    # Extract Frequency Features
    frequencyFeatureDf = feature_extraction(epochSeries=epochSeries, config=yamlConfig)
    # Save frequency features dataframe
    print("Saving frequency feature dataframe...")
    frequencyFeatureDf.to_pickle(os.path.join(mainDir,'generatedData','frequencyFeaturesDf_{}.pkl'.format(label)))
    
    return (epochSeries, frequencyFeatureDf)