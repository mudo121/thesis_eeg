#!/usr/bin/env python

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from typing import List, Dict
import yasa
import numpy as np
from scipy.interpolate import interp1d
import statistics

# Custom Imports
from consts import *
from utils import createChannelList


class DummyTransformer(BaseEstimator, TransformerMixin):
    ''' Custom Transfomer
    
    '''
    def __init__(self):
        pass

    def fit(self, df : pd.DataFrame, y=None):
        return self # nothing else to do

    def transform(self, df : pd.DataFrame):
        return df

class Frequency_Features(BaseEstimator, TransformerMixin):
    ''' Calculate the frequency bands and the bandpower, lower/upper envelope

    - Calculate different frequency bands
    - For each frequency band calculate:
        - bandpower
        - upper envelope of the bandpower
        - lower envelope of the bandpower
    
    - Calculate mean and standrad deviation of band powers and their envlopoes over 5 windows

    If there are 6 Frequency bands this would be a total of 36 Features (6 x 3 x 2)
    6 = frequency bands
    3 = relative bandpower (normal, upper envelope, lower envelope)
    2 = mean and standard deviation of all 3 relative bandpower values
    '''
    
    def __init__(self, samplingRate : int,
                       frequencyBands : list,
                       numberOfChannels : int,
                       kwargsWelch : Dict = dict(average='median', window='hamming'),
                       epochSizeCalculation : int = 5):

        self.samplingRate = samplingRate
        self.frequencyBands = frequencyBands
        self.kwargsWelch = kwargsWelch
        self.epochSizeCalculation = epochSizeCalculation
        self.numberOfChannels = numberOfChannels

        # convert the frequency into the correct format
        self.frequencyBandNames = self.__convertFreqBandsToListAndReturnNames()

        #self.bandpowerMethod = bandpowerMethod

    def __convertFreqBandsToListAndReturnNames(self):
        ''' The yasa bandpower methods expects the frequency bands in a different way than the yaml file can provide
            Therfore we are going to convert it to the correct format

            As well we need later just the frequency names, which we will return here

        E.g. List = [(0.5, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'),
                    (12, 16, 'Sigma'), (16, 30, 'Beta'), (30, 40, 'Gamma')],
        '''

        freqBandNameList = []

        freqBandList = []
        for key, values in self.frequencyBands.items():
            freqBandList.append((values[0], values[1], key))
            freqBandNameList.append(key)

        self.frequencyBands = freqBandList
        print("Frequenccy Bands: {}".format(self.frequencyBands))

        return freqBandNameList


    def fit(self, epochSeries : pd.Series, y=None):
        return self # nothing else to do

    def transform(self, epochSeries : pd.Series) -> pd.DataFrame:
        ''' Calculate the features here '''

        # Calculate the first features, the relative frequency band power
        epochFeatureSeries = self.__calculateRelativeFrequencyBands(epochSeries)
        

        # Create a dict whith channels, frequencies, bandpower and upper & lower envelope
        channelFrequencyDict = self.__createBandpowerUpperLowerEnvelopeDict(epochFeatureSeries = epochFeatureSeries,
                                                                            frequencyBandNames = self.frequencyBandNames,
                                                                            channelList = createChannelList(self.numberOfChannels))


        # calculate mean and std dev of everything
        statisticsBandpowerDict = self.__createStatisticBandpowerDict(channelFrequencyDict = channelFrequencyDict,
                                                                      epochSizeCalculation = self.epochSizeCalculation)


        # make a nice df out of all and return it
        frequencyFeatureDf = self.__createNiceFeatureDf(statisticsBandpowerDict)

        return frequencyFeatureDf

    def __convertDataFrameToNumpyArray(self, df : pd.DataFrame) -> np.ndarray:

        npArrayList = []
        for columnName in df:
            npArrayList.append(df[columnName].to_numpy())

        return np.array(npArrayList)

    def __calculateRelativeFrequencyBands(self, epochSeries : pd.Series) -> pd.Series:
        ''' Calculates the frequecny bands from the given time series (epochSeries) and returns a series of dataframes with the frequency bands

        The Dataframe looks then like this: (depding on the given frequency bands)

        Channel    | Delta | Theta  | Alpha | ...
        --------------------------------------
        channel_1  |  0.53 |  0.323 | ...
        channel_2  |  0.53 |  0.323 | ...
         ....      |  .... | 

        Info for the bandpower function: https://raphaelvallat.com/yasa/build/html/generated/yasa.bandpower.html
        '''
        epochFeatureSeries = []

        for epoch_df in epochSeries:
            
            dataNpArray = self.__convertDataFrameToNumpyArray(epoch_df)

            # calculate the bandpower for this epoch 
            bandpower_df = yasa.bandpower(data=dataNpArray, 
                                        sf=self.samplingRate,
                                        ch_names=epoch_df.columns,
                                        win_sec=4, #  e.g. for a lower frequency of interest of 0.5 Hz, the window length should be at least 2 * 1 / 0.5 = 4 seconds
                                        relative=True, # then the bandpower is already between 0 and 1
                                        bands=self.frequencyBands,
                                        kwargs_welch=self.kwargsWelch) # e.g. (0.5, 4, ‘Delta’)

            
            epochFeatureSeries.append(bandpower_df)
        
        
        return epochFeatureSeries


    def __getEnvelopeModels(self, aTimeSeries, rejectCloserThan = 0):   
        '''Fits models to the upper and lower envelope peaks and troughs.
        
        A peak is defined as a region where the slope transits from positive to negative (i.e. local maximum).
        A trough is defined as a region where the slope transits from negative to positive (i.e. local minimum).
        
        This example uses cubic splines as models.
        
        Parameters:
            aTimeSeries:      A 1 dimensional vector (a list-like).
            rejectCloserThan: An integer denoting the least distance between successive peaks / troughs. Or None to keep all.
        
        
        All Credits to: https://gist.github.com/aanastasiou/480d81361abcdc794783
        
        I added only some conversions for a better future calculations
        '''    
        #Prepend the first value of (s) to the interpolating values. This forces the model to use the same starting point for both the upper and lower envelope models.    
        u_x = [0,]
        u_y = [aTimeSeries[0],]    
        lastPeak = 0
        
        l_x = [0,]
        l_y = [aTimeSeries[0],]
        lastTrough = 0
        
        #Detect peaks and troughs and mark their location in u_x,u_y,l_x,l_y respectively.    
        for k in range(1,len(aTimeSeries)-1):
            #Mark peaks        
            if (np.sign(aTimeSeries[k]-aTimeSeries[k-1])==1) and (np.sign(aTimeSeries[k]-aTimeSeries[k+1])==1) and ((k-lastPeak)>rejectCloserThan):
                u_x.append(k)
                u_y.append(aTimeSeries[k])    
                lastPeak = k
                
            #Mark troughs
            if (np.sign(aTimeSeries[k]-aTimeSeries[k-1])==-1) and ((np.sign(aTimeSeries[k]-aTimeSeries[k+1]))==-1) and ((k-lastTrough)>rejectCloserThan):
                l_x.append(k)
                l_y.append(aTimeSeries[k])
                lastTrough = k
        
        #Append the last value of (s) to the interpolating values. This forces the model to use the same ending point for both the upper and lower envelope models.    
        u_x.append(len(aTimeSeries)-1)
        u_y.append(aTimeSeries[-1])
        
        l_x.append(len(aTimeSeries)-1)
        l_y.append(aTimeSeries[-1])
        
        try:
            #Fit suitable models to the data. Here cubic splines.    
            u_p = interp1d(u_x,u_y, kind = 'cubic',bounds_error = False, fill_value=0.0)
            l_p = interp1d(l_x,l_y,kind = 'cubic',bounds_error = False, fill_value=0.0)
        except ValueError as e:
            # If there is an error, then there are probably too less values to evalutate
            # print(e) -> The number of derivatives at boundaries does not match: expected 1, got 0+0  (That's the Error I had...)
            return (None, None)
        
        # map the result of interp1d to a list
        u_p = list(map(u_p,range(0,len(aTimeSeries))))
        l_p = list(map(l_p,range(0,len(aTimeSeries))))   
        
        # convert the list of ndarray elemts to a list of float elements
        u_p = np.array(u_p, dtype = float)
        l_p = np.array(l_p, dtype = float)
        
        return (u_p,l_p)
    

    def __getValuesFromFeatureEpochs(self, featureEpochSeries : pd.Series, channelName : str, valueName : str):
        ''' Get a list of values from a channel e.g. a list of delta values from channel 1
        
        @param featureEpochSeries: A series of feature epochs
        @param channelName: The channel where we want values from
        @param valueName: The value (column) we want, e.g. Delta or Alpha
        '''
        valueList = []
        for epoch in featureEpochSeries:
            try:
                columnValue = epoch[valueName].filter(items=[channelName], axis='index').iloc[0]
            except IndexError: # If there is no value, set it to NaN
                columnValue = np.nan
                #columnValue = 0

            valueList.append(columnValue)
        return valueList


    def __createBandpowerUpperLowerEnvelopeDict(self, epochFeatureSeries : pd.Series, frequencyBandNames : List[str], channelList: List[str], rejectCloserThan : int = 0) -> Dict[str, Dict[str, Dict[str , List[float]]]]:
        ''' Creates a dict where for each channel and frequency band the bandpower, lower & upper envelope gets calculated and stored in that dict'''
        # A dict of channels, where each channel contains a dict of frequency bands,
        # where this dict contains the bandpower Value List, upper envelope bandpower list, lower envelope bandpower list
        
        print("Creating bandpower, lower & upper envelope dictionary...")
        
        channelFrequencyDict = {}
        
        for channel in channelList:
            
            channelFrequencyDict[channel] = {}
            
            for frequencyBand in frequencyBandNames:
                
                # first get all values from one channel and one frequency band
                bandpowerValueList = self.__getValuesFromFeatureEpochs(featureEpochSeries=epochFeatureSeries, channelName=channel, valueName=frequencyBand)
                
                #Estimate models without rejecting any peak
                P = self.__getEnvelopeModels(bandpowerValueList, rejectCloserThan=rejectCloserThan)        
                #Evaluate each model over the domain of (s)
                bandpowerValueList_upper_envelope = P[0]
                bandpowerValueList_lower_envelope = P[1]
                
                # create a dict for the frequency band
                channelFrequencyDict[channel][frequencyBand] = {}
                
                # update the dict with the bandpower values
                channelFrequencyDict[channel][frequencyBand][INDEX_BANDPOWER_LIST] = bandpowerValueList
                channelFrequencyDict[channel][frequencyBand][INDEX_BANDPOWER_UPPER_ENVELOPE_LIST] = bandpowerValueList_upper_envelope
                channelFrequencyDict[channel][frequencyBand][INDEX_BANDPOWER_LOWER_ENVELOPE_LIST] = bandpowerValueList_lower_envelope
                
        return channelFrequencyDict


    def __createStatisticBandpowerDict(self, channelFrequencyDict : Dict[str, Dict[str, Dict[str , List[int]]]], epochSizeCalculation) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
        ''' Creats a dict which channels and frequency bands, where the mean and standard deviation gets calculated for each bandpower, lower & upper envelope 
        
        The Dict looks then like:
            Dict[channel][frequencyBand][Mean Bandpower] = List of Floats
            Dict[channel][frequencyBand][Mean Bandpower Upper Envelope] = List of Floats
            Dict[channel][frequencyBand][Mean Bandpower Lower Envelope] = List of Floats

            Dict[channel][frequencyBand][Standard Deviation Bandpower] = List of Floats
            Dict[channel][frequencyBand][Standard Deviation Bandpower Upper Envelope] = List of Floats
            Dict[channel][frequencyBand][Standard Deviation Bandpower Lower Envelope] = List of Floats
        '''
        
        print("Creating statistics bandpower dict...")
        
        statisticsBandpowerDict = {}
        
        # loop through channels
        for channel, channelDict in channelFrequencyDict.items():
            
            #print("Creating {} statistics...".format(channel))
            # ##############################
            # create a dict for each channel
            statisticsBandpowerDict[channel] = {}
            
            # loop throgh the frequency bands
            for frequencyBand, frequencyDict in channelDict.items():
                
                #print("Creating {} statistics...".format(frequencyBand))
                
                # ###################################
                # create a dict for each frequency band
                statisticsBandpowerDict[channel][frequencyBand] = {}
                
                # Calculate the mean of the bandpower
                bandpower_mean = self.__calculateMeanOverEpochs(valueList = frequencyDict[INDEX_BANDPOWER_LIST],
                                                        numberOfEpochs = epochSizeCalculation)
                statisticsBandpowerDict[channel][frequencyBand][INDEX_MEAN_BANDPOWER_LIST] = bandpower_mean # append it to the dict

                                                                                
                # Calculate the mean of the upper envelope
                bandpower_upper_envelope_mean = self.__calculateMeanOverEpochs(valueList = frequencyDict[INDEX_BANDPOWER_UPPER_ENVELOPE_LIST],
                                                                        numberOfEpochs = epochSizeCalculation)
                statisticsBandpowerDict[channel][frequencyBand][INDEX_MEAN_BANDPOWER_UPPER_ENVELOPE_LIST] = bandpower_upper_envelope_mean # append it to the dict

                                                                                                
                # Calculate the mean of the lower envelope
                bandpower_lower_envelope_mean = self.__calculateMeanOverEpochs(valueList = frequencyDict[INDEX_BANDPOWER_LOWER_ENVELOPE_LIST],
                                                                        numberOfEpochs = epochSizeCalculation)
                statisticsBandpowerDict[channel][frequencyBand][INDEX_MEAN_BANDPOWER_LOWER_ENVELOPE_LIST] = bandpower_lower_envelope_mean # append it to the dict

                                                                                                
                # ------------------------------------------------------------------------------------------------------------------------
                # Calculate the standard deviation of the mean
                bandpower_std_dev = self.__calculateStandardDeviationOverEpochs(valueList = frequencyDict[INDEX_BANDPOWER_LIST],
                                                                        numberOfEpochs = epochSizeCalculation)
                statisticsBandpowerDict[channel][frequencyBand][INDEX_STD_DEV_BANDPOWER_LIST] = bandpower_std_dev # append it to the dict

                                                                                                
                # Calculate the standard deviation of the upper envelope
                bandpower_upper_envelope_std_dev = self.__calculateStandardDeviationOverEpochs(valueList = frequencyDict[INDEX_BANDPOWER_UPPER_ENVELOPE_LIST],
                                                                        numberOfEpochs = epochSizeCalculation)
                statisticsBandpowerDict[channel][frequencyBand][INDEX_STD_DEV_BANDPOWER_UPPER_ENVELOPE_LIST] = bandpower_upper_envelope_std_dev # append it to the dict

                                                                                                
                # Calculate the standard deviation of the lower envelopes
                bandpower_lower_envelope_std_dev = self.__calculateStandardDeviationOverEpochs(valueList = frequencyDict[INDEX_BANDPOWER_LOWER_ENVELOPE_LIST],
                                                                        numberOfEpochs = epochSizeCalculation)
                statisticsBandpowerDict[channel][frequencyBand][INDEX_STD_DEV_BANDPOWER_LOWER_ENVELOPE_LIST] = bandpower_lower_envelope_std_dev # append it to the dict
        
        
        return statisticsBandpowerDict


    def __calculateMeanOverEpochs(self, valueList : List, numberOfEpochs : int = 5) -> List:
        ''' Calculate the mean over a given number of epochs
            e.g. if numberOfEpoch is 5 then the mean will get calculated from epoch 0-4; 5-9; 10-14; ...
            
        @param valueList: A list of values, where each entry represnts one epoch
        @param numberOfEpochs: A number which defines how many epochs should included in the calculations
        '''
        if valueList is None:
            return None
        
        meanValueList = []
        start = 0
        end = numberOfEpochs
        for i in range(0, len(valueList), numberOfEpochs):
            meanValue = statistics.mean(valueList[start:end])
            meanValueList.append(meanValue)
            start += numberOfEpochs
            end += numberOfEpochs

        return meanValueList

    def __calculateStandardDeviationOverEpochs(self, valueList : List, numberOfEpochs : int = 5) -> List:
        ''' Calculate the standard deviation over a given number of epochs
            e.g. if numberOfEpoch is 5 then the standard deviation will get calculated from epoch 0-4; 5-9; 10-14; ...
            
        @param valueList: A list of values, where each entry represnts one epoch
        @param numberOfEpochs: A number which defines how many epochs should included in the calculations
        '''
        
        if valueList is None:
            return None
        
        stDev_valueList = []
        start = 0
        end = numberOfEpochs
        for i in range(0, len(valueList), numberOfEpochs):
            value = statistics.stdev(valueList[start:end])
            stDev_valueList.append(value)
            start += numberOfEpochs
            end += numberOfEpochs

        return stDev_valueList

    def __createNiceFeatureDf(self, statisticsBandpowerDict : Dict[str, Dict[str, Dict[str, List[float]]]]) -> pd.DataFrame:
        '''  Creates a datframe which includes all features (channels x frequencybands x 3 x 2)
        Each column represents one feature and each row represents one epoch

        A column/feature looks like this:
         {channel}_{frequencyBand}_{feature name}
        '''

        print("Creating a nice feature dataframe...")

        # Create an epoch dataframe
        epochFeatureDf = pd.DataFrame()
        epochFeatureDf.index.name = "Epochs"
        
        # loop through channels
        for channel, channelDict in statisticsBandpowerDict.items():
            
            # loop through the frequency bands
            for frequencyBand, frequencyDict in channelDict.items():
                #print("Freq Dict Keys: {}".format(frequencyDict.keys()))
                
                for feature in FEATURE_DATAFRAME_COLUMNS:
                    
                    columnName = "{ch}_{freq}_{feature}".format(ch=channel, freq=frequencyBand, feature=feature)
                    #print("Feature - {}: {}\n\n".format(feature, frequencyDict[feature]))
    
                    epochFeatureDf[columnName] = frequencyDict[feature]

        return epochFeatureDf
