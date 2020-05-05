#!/usr/bin/env python

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import yasa
import scipy

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

class ResampleSignal(BaseEstimator, TransformerMixin):
    ''' Resample the Signal if neccessary '''
    def __init__(self, samplingRate : int, resampleRate : int):
        self.samplingRate = samplingRate
        self.resampleRate = resampleRate

    def fit(self, df : pd.DataFrame, y=None):
        return self # nothing else to do

    def transform(self, df : pd.DataFrame):
        ''' Resample every column'''
        if self.resampleRate == 0:
            return df

        if self.resampleRate > self.samplingRate:
            raise Exception("Resampling Rate is higher than the actual sampling rate")

        print("Resample Signal from {} to {} Hz".format(self.samplingRate, self.resampleRate))

        sumSamples = len(df.index)
        newSumSamples = int(sumSamples / (self.samplingRate / self.resampleRate))

        resampledDf = pd.DataFrame()

        for column in df:
            resampledDf[column] = scipy.signal.resample(df[column], newSumSamples)

        return resampledDf


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

        elif self.device == DEVICES_MUSE_LSL:
            df['timestamps'] = pd.to_datetime(df['timestamps'], unit='s')
            # set new index and delete the Time125Hz column
            df.set_index(df['timestamps'], inplace=True)
            del df['timestamps']

        elif self.device == DEVICES_NEUROSCAN:
            # We only have to delete those 2 columns
            del df['Unnamed: 0']
            del df['time'] # those are just numbers

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

        elif self.device == DEVICES_MUSE_LSL:
            
            # Delete the right aux and marker column
            df.drop(['Right AUX', 'Marker'], axis=1, inplace=True)

            # And rename the columns to channel 1-4
            df.rename(columns={"TP9": "channel_1", "AF7": "channel_2", "AF8": "channel_3", "TP10" : "channel_4"}, errors="raise", inplace=True)
            return df

        elif self.device == DEVICES_OPEN_BCI:
            # delete unwanted columns (which are not channels)
            for column in df.columns.values:
                if not("channel" in column.lower()):
                    print("Deleting column '{}'".format(column))
                    del df[column]

            return df

        elif self.device == DEVICES_NEUROSCAN:
            
            # generated automatically in jupyter
            renameDict = {'HEOL': 'channel_1', 'HEOR': 'channel_2', 'FP1': 'channel_3', 'FP2': 'channel_4', 'VEOU': 'channel_5', 'VEOL': 'channel_6', 'F7': 'channel_7', 'F3': 'channel_8', 'FZ': 'channel_9', 'F4': 'channel_10', 'F8': 'channel_11', 'FT7': 'channel_12', 'FC3': 'channel_13', 'FCZ': 'channel_14', 'FC4': 'channel_15', 'FT8': 'channel_16', 'T3': 'channel_17', 'C3': 'channel_18', 'CZ': 'channel_19', 'C4': 'channel_20', 'T4': 'channel_21', 'TP7': 'channel_22', 'CP3': 'channel_23', 'CPZ': 'channel_24', 'CP4': 'channel_25', 'TP8': 'channel_26', 'A1': 'channel_27', 'T5': 'channel_28', 'P3': 'channel_29', 'PZ': 'channel_30', 'P4': 'channel_31', 'T6': 'channel_32', 'A2': 'channel_33', 'O1': 'channel_34', 'OZ': 'channel_35', 'O2': 'channel_36', 'FT9': 'channel_37', 'FT10': 'channel_38', 'PO1': 'channel_39', 'PO2': 'channel_40'}
            df.rename(columns=renameDict, errors="raise", inplace=True)
            return df


        else:
            raise NotImplementedError("So far only those are supported: {}".format(SUPPORTED_DEVICES))
    
        return df