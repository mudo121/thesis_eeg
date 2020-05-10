#!/usr/bin/env python

'''
Here a Transformers, which are therer to convert the data into a more general format
'''

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from typing import List
import copy

# cutstom imports
from consts import *


class ExtractSignals(BaseEstimator, TransformerMixin):
    ''' Custom Transformer to extract the raw signals from a given csv file 
    Returns a DataFrame only with raw signals, requires that the Index is already as the timestamp
    '''

    def __init__(self, device : str):
        self.device = device

    def fit(self, df : pd.DataFrame, y=None):
        return self # nothing else to do

    def transform(self, df : pd.DataFrame) -> (pd.DataFrame, List):
        ''' Delete all not necessary columns - returns the remaing columns as a list (the features)'''
        if self.device == DEVICES_MUSE_MONITOR:

            # create a new dataframe only will the 4 raw eeg channels
            new_df = pd.DataFrame()
            new_df['Channel 1'] = df['RAW_TP9'].copy()  # TP9
            new_df['Channel 2'] = df['RAW_AF7'].copy()  # AF7
            new_df['Channel 3'] = df['RAW_AF8'].copy()  # AF8
            new_df['Channel 4'] = df['RAW_TP10'].copy() # TP10
            
            # delete all NaN values, because muse monitor adds extra stuff
            new_df.dropna(inplace=True)

            featureList = ["RAW_TP9", "RAW_AF7", "RAW_AF8", "RAW_TP10"]

            return new_df, featureList

        elif self.device == DEVICES_MUSE_LSL:
            
            # Delete the right aux and marker column
            df.drop(['Right AUX', 'Marker'], axis=1, inplace=True)

            # copy the current columns
            featureList = list(df.columns)

            # And rename the columns to channel 1-4
            df.rename(columns={"TP9": "channel_1", "AF7": "channel_2", "AF8": "channel_3", "TP10" : "channel_4"}, errors="raise", inplace=True)

            return df, featureList

        elif self.device == DEVICES_OPEN_BCI:
            # delete unwanted columns (which are not channels)
            for column in df.columns.values:
                if not("channel" in column.lower()):
                    print("Deleting column '{}'".format(column))
                    del df[column]

            # TODO!
            featureList = []

            return df, featureList

        elif self.device == DEVICES_NEUROSCAN:
            
            # generated automatically in jupyter
            renameDict = {'HEOL': 'channel_1', 'HEOR': 'channel_2', 'FP1': 'channel_3', 'FP2': 'channel_4', 'VEOU': 'channel_5', 'VEOL': 'channel_6', 'F7': 'channel_7', 'F3': 'channel_8', 'FZ': 'channel_9', 'F4': 'channel_10', 'F8': 'channel_11', 'FT7': 'channel_12', 'FC3': 'channel_13', 'FCZ': 'channel_14', 'FC4': 'channel_15', 'FT8': 'channel_16', 'T3': 'channel_17', 'C3': 'channel_18', 'CZ': 'channel_19', 'C4': 'channel_20', 'T4': 'channel_21', 'TP7': 'channel_22', 'CP3': 'channel_23', 'CPZ': 'channel_24', 'CP4': 'channel_25', 'TP8': 'channel_26', 'A1': 'channel_27', 'T5': 'channel_28', 'P3': 'channel_29', 'PZ': 'channel_30', 'P4': 'channel_31', 'T6': 'channel_32', 'A2': 'channel_33', 'O1': 'channel_34', 'OZ': 'channel_35', 'O2': 'channel_36', 'FT9': 'channel_37', 'FT10': 'channel_38', 'PO1': 'channel_39', 'PO2': 'channel_40'}
            df.rename(columns=renameDict, errors="raise", inplace=True)

            featureList = ['HEOL','HEOR','FP1','FP2','VEOU','VEOL','F7','F3','FZ','F4','F8','FT7','FC3','FCZ','FC4','FT8','T3',
                            'C3','CZ','C4','T4','TP7','CP3','CPZ','CP4','TP8','A1','T5','P3','PZ','P4','T6','A2','O1','OZ','O2',
                            'FT9','FT10','PO1','PO2']

            return df, featureList


        else:
            raise NotImplementedError("So far only those are supported: {}".format(SUPPORTED_DEVICES))
    
        return df

class ConvertIndexToTimestamp(BaseEstimator, TransformerMixin):
    ''' Custom Transfomer
    
    Set the timestamp as the index
    '''
    def __init__(self, device, starttime = None):
        self.device = device
        self.starttime = starttime

    def fit(self, df : pd.DataFrame, y=None):
        return self # nothing else to do

    def transform(self, df : pd.DataFrame):
        if self.device == DEVICES_MUSE_MONITOR:
            df = df.set_index('TimeStamp')

        elif self.device == DEVICES_OPEN_BCI:
            
            if self.starttime == None:
                raise RuntimeError("If device is '{}', then you must provide a start time!".format(self.device))

            # convert time:125hz to datetime
            # unit = s - because the timestamp are in seconds
            # origin has to be a timestamp and will set the date where it should start
            df['Time:125Hz'] = pd.to_datetime(df['Time:125Hz'], unit='s', origin=self.starttime)
            
            # set new index and delete the Time125Hz column
            df.set_index(df['Time:125Hz'], inplace=True)
            del df['Time:125Hz']

        elif self.device == DEVICES_MUSE_LSL:
            df['timestamps'] = pd.to_datetime(df['timestamps'], unit='s')
            # set new index and delete the Time125Hz column
            df.set_index(df['timestamps'], inplace=True)
            del df['timestamps']

        elif self.device == DEVICES_NEUROSCAN:
            # We only have to delete those 2 columns
            try:
                del df['Unnamed: 0']
            except Exception as e:
                pass

            try:
                del df['time'] # those are just numbers
            except Exception as e:
                pass

        else:
            raise NotImplementedError("So far only those are supported: {}".format(SUPPORTED_DEVICES))

        return df