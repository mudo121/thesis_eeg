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






