#!/usr/bin/env python

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import yasa
from typing import List
import copy

# cutstom imports
from utils import convertDataFrameToNumpyArray
from consts import *

class DummyTransformer(BaseEstimator, TransformerMixin):
    ''' Custom Transfomer
    
    '''
    def __init__(self):
        pass

    def fit(self, df : pd.DataFrame, y=None):
        return self # nothing else to do

    def transform(self, df : pd.DataFrame):
        return df


class SlidingWindow(BaseEstimator, TransformerMixin):
    ''' Custom Transfomer - creates windows / epochs
    
    Based on this function https://raphaelvallat.com/yasa/build/html/generated/yasa.sliding_window.html
    '''
    def __init__(self, samplingRate, windowSizeInSeconds, overlapInSeconds=None, channelNames=None):
        self.channelNames = channelNames
        self.samplingRate = samplingRate
        self.windowSize = windowSizeInSeconds
        self.overlap = overlapInSeconds

    def fit(self, df : pd.DataFrame, y=None):
        return self # nothing else to do

    def transform(self, df : pd.DataFrame) -> np.ndarray:
        
        print("Creating sliding windows...")

        if self.channelNames == None:
            self.channelNames = self.__returnAllChannelNames(df)

        df_npArrayList = convertDataFrameToNumpyArray(pd=df, channelNames=self.channelNames)

        if self.overlap == None:
            _, data = yasa.sliding_window(df_npArrayList, self.samplingRate, window=self.windowSize)
        else:
            windowStepSize = self.windowSize - self.overlap # calculate the step size of the window
            _, data = yasa.sliding_window(df_npArrayList, self.samplingRate, window=self.windowSize, step=windowStepSize)

        # Data shape (epchos, channels, number of samples)
        return data

    def __returnAllChannelNames(self, pd : pd.DataFrame) -> List[str]:
        channelNames = []
        for column in pd.columns.values:
            if "channel" in column.lower():
                channelNames.append(column)
        return channelNames


class Convert3dArrayToSeriesOfDataframes(BaseEstimator, TransformerMixin):
    ''' Custom Transformer

    Convert a 3D numpy.ndarray to a series of dataframes. It basically just splits up the 3d into 1d and 2d
    '''

    def __init__(self):
        pass

    def fit(self, X , y=None):
        return self # nothing else to do

    def transform(self, data_ndarray : np.ndarray) -> pd.Series:
        # convert it to a series of dataframes
        print("Converting 3d Numpy Array to a series of Df's")

        epochSeries = self.__convert3dArrayToSeriesOfDataframes(data_ndarray)
        return epochSeries

    def __convert3dArrayToSeriesOfDataframes(self, d3Epochs : np.ndarray) -> pd.Series:
        ''' Converts a 3d ndarray to a pandas series filled with dataframe

        The series contains the amount of epochs. Each epoch in the series contains a dataframe with the channels and the actual epoch with the samples
        '''
        channelEpochs = []
        
        for channelEpoch in d3Epochs:
            df = pd.DataFrame()
            channelCounter = 1
            
            for epoch in channelEpoch:
                currentChannel = "channel_{}".format(channelCounter)
                #print(currentChannel)
                df[currentChannel] = epoch
                channelCounter += 1
                
            channelEpochs.append(df)
        
        epochSeries = pd.Series(channelEpochs)
        return epochSeries

    def __convert3DArrayTo3DList(self, epochs : np.ndarray) -> List[List[List]]:
        ''' Converts a 3D numpy.ndarray to a 3D python list 
        It's important to use deepcopy, because deep down list are just pointers :D
        '''
        d3List = []
        
        epochSum, channelSum, sampleSum = np.shape(epochs)
        for r in range(epochSum):
            
            channelEpochs = []
            for i in range(channelSum):
                samples = epochs[r,i,:]
                channelEpochs.append(copy.deepcopy(samples))
            
            
            d3List.append(copy.deepcopy(channelEpochs))
        
        return d3List

class DeleteFaultyEpochs(BaseEstimator, TransformerMixin):
    ''' Custom Transfomer
    
    Delete faulty epochs - A faulty epochs contains e.g. more than 12.5% NaN Values

    Expects a 3D numpy.ndarray but returns a 3D List! Because we can not easily delete single elements in a 3D numpy.ndarray    
    '''
    def __init__(self, maxFaultyRate = 0.15):
        self.maxFaultyRate = maxFaultyRate

    def fit(self, data_ndarray : np.ndarray, y=None):
        return self # nothing else to do

    def transform(self, epochSeries : pd.Series) -> pd.Series:
        ''' Deletes all faulty data from the given array '''
        print("Deleting faulty data...")
        
        # delete faulty epochs inplace
        self.__deleteFaultyEpochs(epochSeries, maxFaultyRate=self.maxFaultyRate)

        return epochSeries

    def __deleteFaultyEpochs(self, epochSeries : pd.Series, maxFaultyRate=0.15):
        ''' Deletes the faulty epochs from the given epochSeries
        A return value is not necceseray because it deletes it inplace ;)
        '''
        faultyCounter = 0
        
        for series_index, df in epochSeries.iteritems(): 
            for df_column, epoch in df.iteritems():
                
                # calculate faulty sparse - the amount of faulty data
                epochFaultySparse = np.isnan(epoch).sum() / epoch.size
                #print(epochFaultySparse)
                
                if epochFaultySparse >= maxFaultyRate:
                    df.drop(columns=[df_column], inplace=True)
                    faultyCounter+=1
                
        #print("Faulty epochs: {}".format(faultyCounter))

    '''
    def __deleteFaultyEpochs(self, epochList : List[List[List]], maxFaultyRate=0.15) -> List[List[List]]: # type list
        faultyEpochsCounter = 0
        
        # create a copy with all epochs - here we can safely delete all faulty epochs and then return it
        updatedEpochList = copy.deepcopy(epochList)
        
        for i in range(0, len(epochList)):
            
            # if a channel epoch gets deleted, we need to adjust the index, because it doesnt adjust itself dynamically
            # but we only need to adjust for deleting!
            deleteIndexAdjustment = 0
            
            for j in range(0, len(epochList[i])):
            
                # calcualte faulty sparse - the amount of faulty data
                epochFaultySparse = np.isnan(epochList[i][j]).sum() / epochList[i][j].size
                
                if epochFaultySparse >= maxFaultyRate:
                    # delete faulty epoch - with adjusted j index
                    del updatedEpochList[i][j-deleteIndexAdjustment]
                    deleteIndexAdjustment += 1 # increase adjustment
                    
                    faultyEpochsCounter+=1
                    #print("Found faulty epoch: {}".format(epochList[i][j]))
        
        print("Faulty epochs: {}".format(faultyEpochsCounter))   
        
        return updatedEpochList
    '''

class NormalizeData(BaseEstimator, TransformerMixin):
    ''' Custom Transfomer
    
    Normalize Data e.g. make the data between -1 and 1
    '''
    def __init__(self):
        pass

    def fit(self, X , y=None):
        return self # nothing else to do

    def transform(self, epochSeries : pd.Series) -> pd.Series:
        print("Normalizing data...")

        normDfList = []
        
        for series_index, df in epochSeries.iteritems():
            # Got this from there
            # https://stackoverflow.com/questions/26414913/normalize-columns-of-pandas-data-frame
            normalized_df=(df-df.min())/(df.max()-df.min()) # scales between 0 and 1
            #normalized_df=(df-df.mean())/df.std() # just scales
            normDfList.append(normalized_df)
        
        normEpochSeries = pd.Series(normDfList)
        return normEpochSeries

class ReplaceNaNs(BaseEstimator, TransformerMixin):
    ''' Custom Transfomer

    Replace all NaN Values with a given value
    '''
    def __init__(self, replaceValue=0):
        self.replaceValue = replaceValue

    def fit(self, data_ndarray : np.ndarray, y=None):
        return self # nothing else to do

    def transform(self, epochSeries : pd.Series):
        print("Deleting Nan's...")
    
        for series_index, df in epochSeries.iteritems(): 
            df.fillna(self.replaceValue, inplace=True)
        
        return epochSeries

