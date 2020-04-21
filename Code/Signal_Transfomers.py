#!/usr/bin/env python

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import yasa

# cutstom imports
from filterFunctions import butter_bandpass_filter
from utils import convertDataFrameToNumpyArray
from consts import *




class CenterData(BaseEstimator, TransformerMixin):
    ''' Custom Transfomer
    
    '''
    def __init__(self):
        pass

    def fit(self, df : pd.DataFrame, y=None):
        return self # nothing else to do

    def transform(self, df : pd.DataFrame):
        
        for column in df.columns.values:
            df[column] = self.__center(df[column])

        return df

    def __center(self, data, mean=None):
        if mean is None:
            mean = np.nanmean(data)    
        return data / mean


class ReplaceOutliers(BaseEstimator, TransformerMixin):
    ''' Custom Transfomer - Replace Outliers
    
    Replaces Outliers with a simple threshold function (values are getting replaced with np.NaN)
    '''
    def __init__(self, device, lowerThreshold, upperThreshold):
        self.device = device
        self.lowerThreshold = lowerThreshold
        self.upperThreshold = upperThreshold

    def fit(self, df : pd.DataFrame, y=None):
        return self # nothing else to do

    def transform(self, df : pd.DataFrame):
        ''' clear faulty noisy data and replace with NaN '''
        
        for column in df.columns.values:
            # for more info about how this is done
            # https://stackoverflow.com/questions/31511997/pandas-dataframe-replace-all-values-in-a-column-based-on-condition

            # delete everything below lower threshold
            df.loc[df[column] < self.lowerThreshold, column] = np.nan # could be set on np.nan - then the filters won't work
            
            # delete everything above upper threshold
            df.loc[df[column] > self.upperThreshold, column] = np.nan # could be set on np.nan - then the filters won't work

        return df



class BandpassFilter(BaseEstimator, TransformerMixin):
    ''' Custom Transfomer - Bandpass filter
    
    Filter the given filter with a butterworth bandpass filter - it will apply the filter to all channels
    '''
    def __init__(self, device, lowcufreq, highcutfreq, samplingRate):
        self.device = device
        self.lowcufreq = lowcufreq
        self.highcutfreq = highcutfreq
        self.samplingRate = samplingRate

    def fit(self, df : pd.DataFrame, y=None):
        return self # nothing else to do

    def transform(self, df : pd.DataFrame):

        # apply filter on all columns
        for column in df.columns.values:                
            dataFiltered = butter_bandpass_filter(data=df[column], lowcut=self.lowcufreq, highcut=self.highcutfreq, fs=self.samplingRate)
            df[column] = dataFiltered

        return df



class BandstopFilter(BaseEstimator, TransformerMixin):
    ''' Custom Transfomer - Bandstop
    
    Bandstopfilter by a butterworth filter
    '''
    def __init__(self, device, lowcufreq, highcutfreq, samplingRate):
        self.device = device
        self.lowcufreq = lowcufreq
        self.highcutfreq = highcutfreq
        self.samplingRate = samplingRate

    def fit(self, df : pd.DataFrame, y=None):
        return self # nothing else to do

    def transform(self, df : pd.DataFrame):

        # apply filter on all columns
        for column in df.columns.values:                
            dataFiltered = butter_bandpass_filter(data=df[column], lowcut=self.lowcufreq, highcut=self.highcutfreq, fs=self.samplingRate, btype='bandstop')
            df[column] = dataFiltered

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

        else:
            raise NotImplementedError("So far only those are supported: {}".format(SUPPORTED_DEVICES))

        return df


class ExtractSignals(BaseEstimator, TransformerMixin):
    ''' Custom Transformer to extract the raw signals from a given csv file 
    Returns a DataFrame only with raw signals, requires that the Index is already as the timestamp
    '''

    def __init__(self, device : str):
        self.device = device

    def fit(self, df : pd.DataFrame, y=None):
        return self # nothing else to do

    def transform(self, df : pd.DataFrame):
        ''' Delete all not necessary columns'''
        if self.device == DEVICES_MUSE_MONITOR:

            # create a new dataframe only will the 4 raw eeg channels
            new_df = pd.DataFrame()
            new_df['Channel 1'] = df['RAW_TP9'].copy()  # TP9
            new_df['Channel 2'] = df['RAW_AF7'].copy()  # AF7
            new_df['Channel 3'] = df['RAW_AF8'].copy()  # AF8
            new_df['Channel 4'] = df['RAW_TP10'].copy() # TP10
            
            # delete all NaN values, because muse monitor adds extra stuff
            new_df.dropna(inplace=True)

            return new_df

        elif self.device == DEVICES_OPEN_BCI:
            # delete unwanted columns (which are not channels)
            for column in df.columns.values:
                if not("channel" in column.lower()):
                    print("Deleting column '{}'".format(column))
                    del df[column]

            return df

        else:
            raise NotImplementedError("So far only those are supported: {}".format(SUPPORTED_DEVICES))
    
        return df