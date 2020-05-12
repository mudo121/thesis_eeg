#!/usr/bin/env python
'''
This file contains function which creates data, ready for machine learning
'''

import pandas as pd
import numpy as np
import os
from typing import List, Dict, Tuple

# local imports
from myDecorators import deprecated

#from myDecorators import deprecated

from utils import readFileCSV, saveFeatureListToFile, loadConfigFile, saveDictToFile
from pipelines import convert_data, filter_signal, pre_process_signal, feature_extraction


def processRawFileWithPipeline(filepath : str, yamlConfig) -> (pd.Series, pd.DataFrame, List[str]):
    ''' Process a given filepath with the current pipelines 
    
    This creates two different data objects:
        epochSeries: This is a panda.Series which contains dataframes. Each index at the series represens one epoch
        frequencyFreatureDf: This is a dataframe of the frequency features of the epochSeries. The index represnts the epochs. The features are the columns
    '''
    if not os.path.isfile(filepath):
        raise Exception("The file '{}' does not exists!".format(filepath))
    
    df = readFileCSV(filepath)
    df, channelNameList =  convert_data(df=df, config=yamlConfig, starttime=None)
    df = filter_signal(df=df, config=yamlConfig) # general filtering
    epochSeries = pre_process_signal(df=df, config=yamlConfig)   # pre-processing
    frequencyFeatureDf = feature_extraction(epochSeries=epochSeries, config=yamlConfig) # extract features
    
    return epochSeries, frequencyFeatureDf, channelNameList


def processRawFileWithPipeline(filepath : str, yamlConfig) -> (pd.Series, pd.DataFrame, List[str]):
    ''' Process a given filepath with the current pipelines 
    
    This creates two different data objects:
        epochSeries: This is a panda.Series which contains dataframes. Each index at the series represens one epoch
        frequencyFreatureDf: This is a dataframe of the frequency features of the epochSeries. The index represnts the epochs. The features are the columns
    '''
    if not os.path.isfile(filepath):
        raise Exception("The file '{}' does not exists!".format(filepath))
    
    df = readFileCSV(filepath)
    df, channelNameList =  convert_data(df=df, config=yamlConfig, starttime=None)
    df = filter_signal(df=df, config=yamlConfig) # general filtering
    epochSeries = pre_process_signal(df=df, config=yamlConfig)   # pre-processing
    frequencyFeatureDf = feature_extraction(epochSeries=epochSeries, config=yamlConfig) # extract features
    
    return epochSeries, frequencyFeatureDf, channelNameList


def safeAndProcessRawFileWithPipeline(rawFilePath : str, fileDir : str, label : str, yamlConfig):
    ''' Process the given rawfilePath and safe the result as pickle files
    This function calls 'processRawFileWithPipeline()' and the two returning data objects will be safed
    
    @param str rawFilePath: path to file which gets process
    @param str fileDir: Directory where the data objects should be stored
    @param str label: A label to know which data we process, e.g. fatigue, normal or awake data
    @param yamlConfig: A loaded yaml config file for processing the data
    '''
    print ("Starting to process {}...".format(rawFilePath))
    # process the file
    epochSeries, frequencyFeatureDf, channelNameList = processRawFileWithPipeline(filepath=rawFilePath, yamlConfig=yamlConfig)
    
    # save the epoch series
    epochSeries.to_pickle(os.path.join(fileDir,'epochSeries_{}.pkl'.format(label)))
    
    # save the frequency df
    frequencyFeatureDf.to_pickle(os.path.join(fileDir,'frequencyFeaturesDf_{}.pkl'.format(label)))
    
    # save the channel name list
    saveFeatureListToFile(featureList=channelNameList,
                          filepath=os.path.join(fileDir, "features_channel_names.txt"))
    
    # save frequency features
    saveFeatureListToFile(featureList=list(frequencyFeatureDf.columns),
                          filepath=os.path.join(fileDir, "features_frequency_df.txt"))


def processRawDatasetToPickleFiles(datasetDirPath : str, device : str, awakeFileName : str,
                                   fatigueFileName : str, normalFileName : str, unlabeledFileName : str):
    '''
    @param str datasetDirPath: Path where the directory of the dataset is
    @param str device: name of the device, to load the correct yaml file for processing
    
    Depending on the dataset there might be awake, normal, fatigue or unlabeled data. 
    @param awakeFileName: filename of the awake data or None then it will be ignored
    @param fatigueFileName: filename of the fatigue data or None then it will be ignored
    @param normalFileName: filename of the normal data or None then it will be ignored
    @param unlabeledFileName: filename of the unlabeled data or None then it will be ignored
    '''
    
    if not os.path.isdir(datasetDirPath):
        raise Exception("The given dir path '{}' does not exist!".format(datasetDirPath))
        
    # Load the yaml config file for the processing
    yamlConfig = loadConfigFile(device)
    
    for root, dirs, files in os.walk(datasetDirPath):
        for subjectDir in dirs:
            print("#############################################")
            print("Process Subject {} Data...".format(subjectDir))
            print("---------------------------------------------")
            
            if awakeFileName is not None: 
                safeAndProcessRawFileWithPipeline(rawFilePath=os.path.join(root, subjectDir, awakeFileName),
                                                  fileDir=os.path.join(root, subjectDir),
                                                  label = "awake",
                                                  yamlConfig=yamlConfig)
                
            if fatigueFileName is not None: 
                safeAndProcessRawFileWithPipeline(rawFilePath=os.path.join(root, subjectDir, fatigueFileName),
                                                  fileDir=os.path.join(root, subjectDir),
                                                  label = "fatigue",
                                                  yamlConfig=yamlConfig)
                
            if normalFileName is not None: 
                safeAndProcessRawFileWithPipeline(rawFilePath=os.path.join(root, subjectDir, normalFileName),
                                                  fileDir=os.path.join(root, subjectDir),
                                                  label = "normal",
                                                  yamlConfig=yamlConfig)
                
            if unlabeledFileName is not None: 
                safeAndProcessRawFileWithPipeline(rawFilePath=os.path.join(root, subjectDir, normalFileName),
                                                  fileDir=os.path.join(root, subjectDir),
                                                  label = "unlabeled",
                                                  yamlConfig=yamlConfig)
    
    print("#######################################")
    print("Done processing and saving a complete Dataset!")


def loadPickeldData(dataDir : str, label : str):
    ''' Load the epochseries and frequency feature df
    
    @param str dataDir: Directory where the data is
    @param str label: decide which 
    '''
    try:
        epochSeries = pd.read_pickle(os.path.join(dataDir,'epochSeries_{}.pkl'.format(label)))
    except Exception as e:
        #print (e)
        epochSeries = None
        
    try:
        frequencyFeatureDf = pd.read_pickle(os.path.join(dataDir,'frequencyFeaturesDf_{}.pkl'.format(label)))
    except Exception as e:
        #print (e)
        frequencyFeatureDf = None

    return epochSeries, frequencyFeatureDf


def loadPickeldDataset(datasetDirPath : str) -> Dict:
    ''' This functions loads a complete dataset into a dict
    
    Each subject should have a folder with a number in that dict.
    E.g. 1 for subject 1, 2 for subject 2, etc.
    So the datasetDir has only folders with numbers for each subject

    Each Subject contains a dict with 'awake', 'normal', 'fatigue' and 'unlabeled' entry.
    Each entry contain the epochSeries and frequencyFeatureDf
    '''
    
    if not os.path.isdir(datasetDirPath):
        raise Exception("The given dir path '{}' does not exist!".format(datasetDirPath))
    
    datasetDict = {}
    
    for root, dirs, files in os.walk(datasetDirPath):
        for subjectDir in dirs:
            print("Load Subject {} Data...".format(subjectDir))
            
            epochSeries_awake, frequencyFeatureDf_awake = loadPickeldData(dataDir = os.path.join(datasetDirPath, subjectDir),
                                                                          label="awake")
            
            epochSeries_normal, frequencyFeatureDf_normal = loadPickeldData(dataDir = os.path.join(datasetDirPath, subjectDir),
                                                                          label="normal")
            
            epochSeries_fatigue, frequencyFeatureDf_fatigue = loadPickeldData(dataDir = os.path.join(datasetDirPath, subjectDir),
                                                                          label="fatigue")
            
            epochSeries_unlabeled, frequencyFeatureDf_unlabeled = loadPickeldData(dataDir = os.path.join(datasetDirPath, subjectDir),
                                                                          label="unlabeled")
            
            datasetDict[subjectDir] = {"awake" : (epochSeries_awake, frequencyFeatureDf_awake),
                                       "normal" : (epochSeries_normal, frequencyFeatureDf_normal),
                                       "fatigue" : (epochSeries_fatigue, frequencyFeatureDf_fatigue),
                                       "unlabeled" : (epochSeries_unlabeled, frequencyFeatureDf_unlabeled)}
    return datasetDict



def createXyFromDataSeries(dataSeries : pd.Series, target : int) -> (np.ndarray, np.ndarray):
    ''' Create X and y for machine learning
    
    @param pd.Series dataSeries: Should be a series of dataframes
    
    X should look like this [samples, timesteps, features]
        samples: The epoch
        timesteps: E.g. if the epoch contains 200 values then the timestep should contain 200 values
        features: The actual value
    
    y should look tlike this [classIds] 
        classIds: The label for the sample of the X Data
    '''
    
    samples = []
    targetArray = []
    
    if dataSeries is None:
        raise TypeError("Data Series is None!")
    
    if type(dataSeries) != pd.Series:
        raise Exception("The given dataSeries is not a pd.Series! It is {}".format(type(dataSeries)))
    
    # loop through the data Series
    for df in dataSeries:
        
        if type(df) != pd.DataFrame: # check the type
            raise Exception("The dataseries contains a {} object - The series should dataframes only!".format(type(df)))
            
        timesteps = []
            
        for index, row in df.iterrows():
            features = row.to_numpy() # features
            timesteps.append(features)
        
        samples.append(timesteps)
        targetArray.append(target)
    
    
    X = np.array(samples)
    y = np.array(targetArray)
    
    return X, y


def createMachineLearningDataset(eegDataset, targetLabelDict) -> (np.array, np.array):
    
    X = None
    y = None
    
    for subject in eegDataset:
        for target in targetLabelDict:
            
            try:
                print("Processing Subject {} - Target: {} ...".format(subject, target))
                eegDataset[subject][target][0]
                tempX, tempy = createXyFromDataSeries(dataSeries = eegDataset[subject][target][0],
                                       target = targetLabelDict[target])

                if X is None:
                    X = tempX
                else:
                    X = np.concatenate((X, tempX))

                if y is None:
                    y = tempy
                else:
                    y = np.concatenate((y, tempy))
            
            except TypeError:
                print("Skipping Target: {}".format(target))
    
    print("Done!")
    return X,y


def createAndSafeMlDataset(eegDataset : Dict[str, Dict[str ,Tuple[pd.Series, pd.DataFrame]]], targetLabelDict : Dict,
                           dirPath : str) -> (np.array, np.array):
    
    if not os.path.isdir(dirPath):
        raise Exception("The given directory path is not valid! Given path: {}".format(dirPath))
    
    print("Creating Machine Learning Dataset!")
    X, y = createMachineLearningDataset(eegDataset, targetLabelDict)
    
    #Todo - Build the feature df somehow into that too
    
    print("\nSaving Machine Learning Dataset into this directory: {}".format(dirPath))
    np.save(os.path.join(dirPath, "data_series_X.npy"), X)
    np.save(os.path.join(dirPath, "data_series_y.npy"), y)
    
    # Save target labels
    saveDictToFile(targetLabelDict, filepath=os.path.join(dirPath,'target_labels.txt'))
    
    return X, y

@deprecated("Rather use the other functions here to create a feature df")
def generateFeatureDf(csvFilePath, starttime, yamlConfig, label : str, generateData : bool = True, mainDir : str = None) -> (pd.Series, pd.DataFrame):
    ''' Generates dataframes of given csv filepaths
    These dataframes are containg features of the eeg data.

    It returns the series of epoch (of the eeg data) and the feature df
    '''
    
    if not generateData:
        # Load the data
        if mainDir is None:
            print("Enter a main directory where the pickle files can be found!")
            return None
        print ("Reading pickles files for {} data...".format(label))
        frequencyFeatureDf = pd.read_pickle(os.path.join(mainDir,'frequencyFeaturesDf_{}.pkl'.format(label)))
        epochSeries = pd.read_pickle(os.path.join(mainDir,'epochSeries_{}.pkl'.format(label)))
        return (epochSeries, frequencyFeatureDf)
        
    #######################
    #### GENERATE DATA ####
    # ---------------------
    print ("Generating {} driving feature df...".format(label))
    # load dataset
    df = readFileCSV(csvFilePath)  
    
    # Filter the signal
    df = filter_signal(df=df, config=yamlConfig, starttime=starttime)

    # Pre-process the signal
    epochSeries = pre_process_signal(df=df, config=yamlConfig)
    # Save epochSeries
    print("Saving epoch series...")
    epochSeries.to_pickle(os.path.join(mainDir,'epochSeries_{}.pkl'.format(label)))


    # Extract Frequency Features
    frequencyFeatureDf = feature_extraction(epochSeries=epochSeries, config=yamlConfig)
    # Save frequency features dataframe
    print("Saving frequency feature dataframe...")
    frequencyFeatureDf.to_pickle(os.path.join(mainDir,'frequencyFeaturesDf_{}.pkl'.format(label)))
    
    return (epochSeries, frequencyFeatureDf)





