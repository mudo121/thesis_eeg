from sklearn.pipeline import Pipeline
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# CUSTOM IMPORTS
from CustomTransfomers import (ConvertIndexToTimestamp, ExtractSignals,
                                BandpassFilter, BandstopFilter, ReplaceOutliers,
                                CenterData)
from consts import *

def readFileCSV(filePath : str) -> pd.DataFrame:
    df = pd.read_csv(filePath)
    return df


def testMuseData():
    currentDevice = DEVICES_MUSE_MONITOR
    samplingRate = 256
    lowcutFreq = 0.1
    highcutFreq = 60

    # replace outliers
    upperThreshold = 200
    lowerThreshold = -200

    # Create a pipeline for the Muse Monitor - the order is very important!
    signal_processing_pipeline = Pipeline([
        ('Convert Index to Timestamp', ConvertIndexToTimestamp(device=currentDevice)),
        ('Extract Signals', ExtractSignals(device=currentDevice)),
        ('Bandpass Filter', BandpassFilter(device=currentDevice, lowcufreq=lowcutFreq, highcutfreq=highcutFreq, samplingRate=samplingRate)),
        ('Bandstop Filter', BandstopFilter(device=currentDevice, lowcufreq=49, highcutfreq=51, samplingRate=samplingRate))
        #('Replace Outliers', ReplaceOutliers(device=currentDevice, lowerThreshold=lowerThreshold, upperThreshold=upperThreshold))
    ])

    df = readFileCSV("D:/OneDrive - bwedu/Masterthesis/Code/museMonitor_2017-07-25--13-53-06 - Coding.csv")
    df = signal_processing_pipeline.fit_transform(df)

    print(df.head())
    

    plt.show(df.plot())


def testOpenBciData():
    currentDevice = DEVICES_OPEN_BCI
    samplingRate = 125
    lowcutFreq = 0.1
    highcutFreq = 60

    # replace outliers
    upperThreshold = 200
    lowerThreshold = -200


    dataFilepath = "D:/OneDrive - bwedu/Masterthesis/Experiments+Data/Fahren+AimLab/2020_03_05_Propand_1/openBci_record-[2020.03.05-12.27.35]_raw_awake_aimlab.csv"
    df = readFileCSV(dataFilepath)  
    starttime = pd.Timestamp(datetime.strptime('[2020.03.05-12.27.27]', "[%Y.%m.%d-%H.%M.%S]"))

    # Create a pipeline for the Muse Monitor - the order is very important!
    signal_processing_pipeline = Pipeline([
        ('Convert Index to Timestamp', ConvertIndexToTimestamp(device=currentDevice, starttime=starttime)),
        ('Extract Signals', ExtractSignals(device=currentDevice)),
        ('Bandpass Filter', BandpassFilter(device=currentDevice, lowcufreq=lowcutFreq, highcutfreq=highcutFreq, samplingRate=samplingRate)),
        ('Bandstop Filter', BandstopFilter(device=currentDevice, lowcufreq=49, highcutfreq=51, samplingRate=samplingRate)),
        ('Replace Outliers', ReplaceOutliers(device=currentDevice, lowerThreshold=lowerThreshold, upperThreshold=upperThreshold))
    ])

    

    df = signal_processing_pipeline.fit_transform(df)

    print(df.head())

    plt.show(df.plot())

def main():

    #testMuseData()

    testOpenBciData()




if __name__ == "__main__":
    main()