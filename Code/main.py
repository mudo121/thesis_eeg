#!/usr/bin/env python

from sklearn.pipeline import Pipeline
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import yaml
import os
from typing import Dict

# CUSTOM IMPORTS
from Signal_Transfomers import (ConvertIndexToTimestamp, ExtractSignals,
                                BandpassFilter, BandstopFilter, ReplaceOutliers,
                                CenterData)

from Pre_Processing_Transformers import (SlidingWindow, NormalizeData, DeleteFaultyEpochs,
                                            ReplaceNaNs)

from plotFunctions import plotInteractiveEpochs


from consts import *



# ------------ TODO's -------------------------------
#TODO Implement Grid search to find good hyperparameters
# -------------------------------------------------------

def loadConfigFile(configFilePath : str) -> Dict:
    with open(configFilePath, 'r') as stream:
        try:
            yamlConfig = yaml.safe_load(stream)
            return yamlConfig
        except yaml.YAMLError as exc:
            print(exc)
            return None

def readFileCSV(filePath : str) -> pd.DataFrame:
    df = pd.read_csv(filePath)
    return df


def filter_signal(df : pd.DataFrame, config : Dict, starttime=None) -> pd.DataFrame:
    ''' Filter the signal with bandpass, bandstopp and repace outliers '''

    # signal processing pipeline - the first pipeline - e.g. extract the signal from the raw .csv and filter it
    signal_processing_pipeline = Pipeline([
        ('Convert Index to Timestamp', ConvertIndexToTimestamp(device=config['deviceName'], starttime=starttime)),
        ('Extract Signals', ExtractSignals(device=config['deviceName'])),
        ('Bandpass Filter', BandpassFilter(device=config['deviceName'], lowcufreq=config['lowcutFreq_bandpass'], highcutfreq=config['highcutFreq_bandpass'], samplingRate=config['samplingRate'])),
        ('Bandstop Filter', BandstopFilter(device=config['deviceName'], lowcufreq=config['lowcutFreq_bandstopp'], highcutfreq=config['highcutFreq_bandstopp'], samplingRate=config['samplingRate'])),
        ('Replace Outliers', ReplaceOutliers(device=config['deviceName'], lowerThreshold=config['lowerThreshold'], upperThreshold=config['upperThreshold']))
    ])
    df = signal_processing_pipeline.fit_transform(df)
    return df

def pre_process_signal(df : pd.DataFrame, config : Dict) -> pd.Series:
    ''' Pre-process the signal by creating epochs, delete faulty epochs and normalize it
    
    Returns a series of dataframes
    '''
    # pre-process the pipeline for machine learning
    pre_processing_pipeline = Pipeline([
        ('Create Epochs', SlidingWindow(samplingRate=config['samplingRate'], windowSizeInSeconds=config['epochWindowSize'], overlapInSeconds=config['overlap'])),
        ('Delete Faulty Epochs', DeleteFaultyEpochs(maxFaultyRate=config['maxFaultyRate'])), # returns a numpy series with dataframes
        ('Replace NaNs with Zero', ReplaceNaNs()),
        ('Normalize Data', NormalizeData())
    ])

    epochSeries = pre_processing_pipeline.fit_transform(df)
    return epochSeries

def feature_extraction(epochSeries : pd.Series) -> epochSeries:
    ''' Extract features from the generated epoch series 
    
    Do the whole feature extraction in a pipeline
    '''

    feature_extraction_pipeline = Pipeline([
        ("", )
    ])

    epochSeries = feature_extraction_pipeline.fit_transform(epochSeries)

    return epochSeries


def main():

    GENERATE_DATA = False
    # load config
    mainDir = "D:/Masterthesis/thesis_eeg/"

    configFilePath = "D:/Masterthesis/thesis_eeg/config/openBci.yaml"
    yamlConfig = loadConfigFile(configFilePath)
    print(yamlConfig)
    
    if GENERATE_DATA:
        print("Generating data...")

        # load dataset
        dataFilepath = "D:/OneDrive - bwedu/Masterthesis/Experiments+Data/Fahren+AimLab/2020_03_05_Propand_1/openBci_record-[2020.03.05-12.27.35]_raw_awake_aimlab.csv"
        df = readFileCSV(dataFilepath)  
        starttime = pd.Timestamp(datetime.strptime('[2020.03.05-12.27.27]', "[%Y.%m.%d-%H.%M.%S]"))

        # Filter the signal
        df = filter_signal(df=df, config=yamlConfig, starttime=starttime)

        # Pre-process the signal
        epochSeries = pre_process_signal(df=df, config=yamlConfig)

        # Save epochSeries
        print("Saving epoch series...")
        #epochSeries.to_csv(os.path.join(mainDir,'generatedData','epochSeries.csv'))
        epochSeries.to_pickle(os.path.join(mainDir,'generatedData','epochSeries.pkl'))

    else:
        # load data
        print("Loading data...")
        epochSeries = pd.read_pickle(os.path.join(mainDir,'generatedData','epochSeries.pkl'))
    


    print(epochSeries[0].head())
    plt.show(epochSeries[0].plot())

    #plotInteractiveEpochs(epochSeries)

if __name__ == "__main__":
    main()