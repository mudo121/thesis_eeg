#!/usr/bin/env python

# Created: 20.04.2020
# Functions for plotting the data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

def createSampleEpochSeries() -> pd.Series:
    epoch_1_df = pd.DataFrame()
    epoch_1_df['channel_1'] = np.arange(0,9)
    epoch_1_df['channel_2'] = np.arange(10,19)
    epoch_1_df['channel_3'] = np.arange(40,49)

    epoch_2_df = pd.DataFrame()
    epoch_2_df['channel_1'] = np.arange(2,11)
    epoch_2_df['channel_2'] = np.arange(20,29)
    epoch_2_df['channel_3'] = np.arange(50,59)

    epoch_3_df = pd.DataFrame()
    epoch_3_df['channel_1'] = np.arange(4,13)
    epoch_3_df['channel_2'] = np.arange(30,39)
    epoch_3_df['channel_3'] = np.arange(60,69)

    channelEpochs = [epoch_1_df, epoch_2_df, epoch_3_df]
    epochSeries = pd.Series(channelEpochs)

    return epochSeries

def plotInteractiveEpochs(epochSeries : pd.Series = None):
    ''' Create an interactive plot to play around with the epochs '''

    if epochSeries == None:
        epochSeries = createSampleEpochSeries()


    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)

    for column in epochSeries[2]:
        plt.plot(epochSeries[2].index, epochSeries[2][column])

    
    #l, = plt.plot(epochSeries[0].index, epochSeries[0].all(), lw=2, color="red")
    #l = epochSeries[0].plot()
    #plt.axis([0, 1, -10, 10])

    # add slider
    axcolor = 'lightgoldenrodyellow'
    axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)


    epochSlider = Slider(axfreq, 'epochs', 0, len(epochSeries)-1, valinit=0, valstep=1.0, )

    def epochSliderChanged(epoch):

        for axLines in ax:
            axLines.remove()

        epoch = int(epoch)
        for column in epochSeries[epoch]:
            plt.plot(epochSeries[epoch].index, epochSeries[epoch][column])
            
        
        plt.draw()
        plt.pause(0.01)

        #fig.canvas.draw_idle()
    
    epochSlider.on_changed(epochSliderChanged)
    

    plt.show()


def plotFeatureEpochs(featureEpochs : pd.Series, columnsToUse = None, channelsToUse = None):
    ''' Plot features for a complete series
    
    @param featureEpochs: A series of dataframes where the features are stored
    @param columnsToUse: A list of columns to show - If None use all
    @parm channelsToUse: A list of channels to show - If None use all
    
    Example: 
        channelsToUse = ["channel_17", "channel_18"]
        columnsToUse = ['Delta', 'Alpha', 'Theta','Beta','Gamma', 'TotalAbsPow']
        plotFeatureEpochs(epochSeriesFeatures, columnsToUse=columnsToUse, channelsToUse=channelsToUse)
    '''
    
    # the x-line is just the number of epochs (for now)
    xValues = np.arange(0, len(featureEpochs))
    
    # Use a dict to store multiples dicts with lists
    yChannelValuesDict = {}
    
    # init the dict with empty dicts which are containting empty lists
    for channel in channelsToUse:
        yChannelValuesDict[channel] = {}
        for column in columnsToUse:
            yChannelValuesDict[channel][column] = []
    
    
    for epoch in featureEpochs: # loop through all epochs
        for column in columnsToUse: # loop through all columns we want to show
            for channel in channelsToUse: # loop thorugh all channels
                try:
                    columnValue = epoch[column].filter(items=[channel], axis='index').iloc[0]
                except IndexError: # If there is no value, set it to NaN
                    columnValue = np.nan
                yChannelValuesDict[channel][column].append(columnValue) # append the column value and re-assign it
    
    
    
    # plot all yValues
    
    for channel, channelValueDict in yChannelValuesDict.items():
        plt.figure(figsize=(20,5))
        for columnName, columnValueList in channelValueDict.items():
            plt.plot(xValues, columnValueList)
        
        plt.legend(columnsToUse, loc="upper left")
        plt.title("{ch} - Bandpower over all Epochs".format(ch=channel))
        plt.ylabel("Relative power spectral density")
        plt.xlabel("Epochs")

if __name__ == "__main__":
    plotInteractiveEpochs()