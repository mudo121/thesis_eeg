#!/usr/bin/env python
'''
This file contains functinos where pipelines are defined. E.g. a pre-processing pipeline
'''

import pandas as pd
from typing import Dict
from sklearn.pipeline import Pipeline
import os

# Local imports
from Transfomer_Signal import (ConvertIndexToTimestamp, ExtractSignals,
                                BandpassFilter, BandstopFilter, ReplaceOutliers,
                                CenterData, ResampleSignal)
from Transformer_Pre_Processing import (SlidingWindow, NormalizeData, DeleteFaultyEpochs,
                                            ReplaceNaNs)
from Transformer_Feature_Extraction import (Frequency_Features)
from Measuring_Functions import (getChannelUsageInEpochSeries)
from plotFunctions import (plotInteractiveEpochs, plotFeatureEpochs)
from utils import readFileCSV


def filter_signal(df : pd.DataFrame, config : Dict, starttime=None) -> pd.DataFrame:
    ''' Filter the signal with bandpass, bandstopp and repace outliers '''

    # signal processing pipeline - the first pipeline - e.g. extract the signal from the raw .csv and filter it
    signal_processing_pipeline = Pipeline([
        ('Convert Index to Timestamp', ConvertIndexToTimestamp(device=config['deviceName'], starttime=starttime)),
        ('Extract Signals', ExtractSignals(device=config['deviceName'])),
        ('Resample Signal', ResampleSignal(samplingRate=config['samplingRate'], resampleRate=config['resampleRate'])),
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

def feature_extraction(epochSeries : pd.Series, config : Dict) -> pd.Series:
    ''' Extract features from the generated epoch series 
    
    Do the whole feature extraction in a pipeline
    '''

    feature_extraction_pipeline = Pipeline([
        ("Frequency Band feature extraction", Frequency_Features(samplingRate=config['samplingRate'], frequencyBands=config['frequencyBands'],
                                                                numberOfChannels=config['numberOfChannels'], epochSizeCalculation=config['epochSizeCalculation']))
    ])

    epochSeries = feature_extraction_pipeline.fit_transform(epochSeries)

    return epochSeries

