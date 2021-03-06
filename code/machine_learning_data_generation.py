#!/usr/bin/env python
'''
This file contains function which creates data, ready for machine learning
'''

import pandas as pd
import numpy as np
import os
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


# local imports
from myDecorators import deprecated
from utils import readFileCSV, saveFeatureListToFile, loadConfigFile, saveDictToFile
from pipelines import convert_data, filter_signal, prepare_signal, feature_extraction
from consts import TARGET_AWAKE, TARGET_FATIGUE, TARGET_NORMAL, TARGET_UNLABELED


def processRawFileWithPipeline(filepath : str, yamlConfig) -> (pd.Series, pd.DataFrame, List[str], np.ndarray, List[str]):
    ''' Process a given filepath with the current pipelines
    
    This creates three different data objects:
        epochSeries: This is a panda.Series which contains dataframes. Each index at the series represens one epoch
        frequencyFreatureDf: This is a dataframe of the frequency features of the epochSeries. The index represnts the epochs. The features are the columns
        entropyArray: There is the entropy data stored
    '''
    if not os.path.isfile(filepath):
        raise Exception("The file '{}' does not exists!".format(filepath))
    
    df = readFileCSV(filepath)
    df, channelNameList =  convert_data(df=df, config=yamlConfig, starttime=None)
    df = filter_signal(df=df, config=yamlConfig) # general filtering
    epochSeries = prepare_signal(df=df, config=yamlConfig)   # pre-processing
    frequencyFeatureDf, entropyArray, entropyFeatureList = feature_extraction(epochSeries=epochSeries, config=yamlConfig) # extract features
    
    return epochSeries, frequencyFeatureDf, channelNameList, entropyArray, entropyFeatureList


def safeAndProcessRawFileWithPipeline(rawFilePath : str, fileDir : str, label : str, yamlConfig, extraLabel : str = None):
    ''' Process the given rawfilePath and safe the result as pickle files
    This function calls 'processRawFileWithPipeline()' and the two returning data objects will be safed
    
    @param str rawFilePath: path to file which gets process
    @param str fileDir: Directory where the data objects should be stored
    @param str label: A label to know which data we process, e.g. fatigue, normal or awake data
    @param yamlConfig: A loaded yaml config file for processing the data
    @param str extraLabel: Can be used to add an extra text to the filenames. E.g. the source of the data (P300, concentration game, driving...)
    '''
    print ("Starting to process {}...".format(rawFilePath))
    # process the file
    epochSeries, frequencyFeatureDf, channelNameList, entropyArray, entropyFeatureList = processRawFileWithPipeline(filepath=rawFilePath, yamlConfig=yamlConfig)
    
    # save the epoch series
    fileName = 'epochSeries_{}_{}.pkl'.format(extraLabel, label) if extraLabel else 'epochSeries_{}.pkl'.format(label)
    epochSeries.to_pickle(os.path.join(fileDir, fileName))
    
    # save the frequency df
    fileName = 'frequencyFeaturesDf_{}_{}.pkl'.format(extraLabel, label) if extraLabel else 'frequencyFeaturesDf_{}.pkl'.format(label)
    frequencyFeatureDf.to_pickle(os.path.join(fileDir, fileName))
    
    # save the channel name list
    fileName = 'features_channel_names_{}.txt'.format(extraLabel) if extraLabel else 'features_channel_names.txt'
    saveFeatureListToFile(featureList=channelNameList,
                          filepath=os.path.join(fileDir, fileName))
    
    # save frequency features
    fileName = 'features_frequency_df_{}.txt'.format(extraLabel) if extraLabel else 'features_frequency_df.txt'
    saveFeatureListToFile(featureList=list(frequencyFeatureDf.columns),
                          filepath=os.path.join(fileDir, fileName))

    # Save entropy array (not a pickle file)
    fileName = 'entropyFeatures_{}_{}.npy'.format(extraLabel, label) if extraLabel else 'entropyFeatures_{}.npy'.format(label)
    np.save(os.path.join(fileDir, fileName), entropyArray)

    # Save entropy feature list
    fileName = 'features_entropy_{}.txt'.format(extraLabel) if extraLabel else 'features_entropy.txt'
    saveFeatureListToFile(featureList=entropyFeatureList,
                          filepath=os.path.join(fileDir, fileName))


def processRawDatasetToPickleFiles(datasetDirPath : str, device : str, awakeFileName : str,
                                   fatigueFileName : str, normalFileName : str, unlabeledFileName : str,
                                   skipDirs : List[str], extraLabel : str = None):
    '''
    @param str datasetDirPath: Path where the directory of the dataset is
    @param str device: name of the device, to load the correct yaml file for processing
    
    Depending on the dataset there might be awake, normal, fatigue or unlabeled data. 
    @param awakeFileName: filename of the awake data or None then it will be ignored
    @param fatigueFileName: filename of the fatigue data or None then it will be ignored
    @param normalFileName: filename of the normal data or None then it will be ignored
    @param unlabeledFileName: filename of the unlabeled data or None then it will be ignored
    @param str extraLabel: If not None, then this will be added to every file, e.g. the source of the data (P300, game, driving, ...)
    '''
    
    if not os.path.isdir(datasetDirPath):
        raise Exception("The given dir path '{}' does not exist!".format(datasetDirPath))
        
    # Load the yaml config file for the processing
    yamlConfig = loadConfigFile(device)
    
    for root, dirs, files in os.walk(datasetDirPath):
        for subjectDir in dirs:

            if subjectDir not in skipDirs:

                print("#############################################")
                print("Process Subject {} Data...".format(subjectDir))
                print("---------------------------------------------")
                
                if awakeFileName is not None: 
                    safeAndProcessRawFileWithPipeline(rawFilePath=os.path.join(root, subjectDir, awakeFileName),
                                                    fileDir=os.path.join(root, subjectDir),
                                                    label = "awake",
                                                    yamlConfig=yamlConfig,
                                                    extraLabel=extraLabel)
                    
                if fatigueFileName is not None: 
                    safeAndProcessRawFileWithPipeline(rawFilePath=os.path.join(root, subjectDir, fatigueFileName),
                                                    fileDir=os.path.join(root, subjectDir),
                                                    label = "fatigue",
                                                    yamlConfig=yamlConfig,
                                                    extraLabel=extraLabel)
                    
                if normalFileName is not None: 
                    safeAndProcessRawFileWithPipeline(rawFilePath=os.path.join(root, subjectDir, normalFileName),
                                                    fileDir=os.path.join(root, subjectDir),
                                                    label = "normal",
                                                    yamlConfig=yamlConfig,
                                                    extraLabel=extraLabel)
                    
                if unlabeledFileName is not None: 
                    safeAndProcessRawFileWithPipeline(rawFilePath=os.path.join(root, subjectDir, unlabeledFileName),
                                                    fileDir=os.path.join(root, subjectDir),
                                                    label = "unlabeled",
                                                    yamlConfig=yamlConfig,
                                                    extraLabel=extraLabel)
            
            else:
                print(" *********** Skipping Directory '{}' ***********".format(subjectDir))
    
    print("#######################################")
    print("Done processing and saving a complete Dataset!")


def loadPickeldData(dataDir : str, label : str, epochSeries : str = 'epochSeries_{}.pkl', frequencyData : str = 'frequencyFeaturesDf_{}.pkl', entropyData : str = 'entropyFeatures_{}.npy'):
    ''' Load the epochseries and frequency feature df
    
    @param str dataDir: Directory where the data is
    @param str label: decide which
    '''
    try:
        epochSeries = pd.read_pickle(os.path.join(dataDir, epochSeries.format(label.lower())))
    except Exception as e:
        #print (e)
        epochSeries = None
        
    try:
        frequencyFeatureDf = pd.read_pickle(os.path.join(dataDir, frequencyData.format(label.lower())))
    except Exception as e:
        #print (e)
        frequencyFeatureDf = None

    try:
        entropy_3d_array = np.load(os.path.join(dataDir, entropyData.format(label.lower())))
    except Exception as e:
        entropy_3d_array = None

    return epochSeries, frequencyFeatureDf, entropy_3d_array


def loadPickeldDataset(datasetDirPath:str, skipDirs : List[str]) -> Dict:
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

            if subjectDir not in skipDirs:
                print("Load Subject {} Data...".format(subjectDir))
            
                epochSeries_awake, frequencyFeatureDf_awake, entropy_3d_array_awake = loadPickeldData(dataDir = os.path.join(datasetDirPath, subjectDir),
                                                                                                      label=TARGET_AWAKE)
                
                epochSeries_normal, frequencyFeatureDf_normal, entropy_3d_array_normal = loadPickeldData(dataDir = os.path.join(datasetDirPath, subjectDir),
                                                                                                label=TARGET_NORMAL)
                
                epochSeries_fatigue, frequencyFeatureDf_fatigue, entropy_3d_array_fatigue = loadPickeldData(dataDir = os.path.join(datasetDirPath, subjectDir),
                                                                                                    label=TARGET_FATIGUE)
                
                epochSeries_unlabeled, frequencyFeatureDf_unlabeled, entropy_3d_array_unlabeled = loadPickeldData(dataDir = os.path.join(datasetDirPath, subjectDir),
                                                                                                        label=TARGET_UNLABELED)
                
                datasetDict[subjectDir] = {TARGET_AWAKE : (epochSeries_awake, frequencyFeatureDf_awake, entropy_3d_array_awake),
                                        TARGET_NORMAL : (epochSeries_normal, frequencyFeatureDf_normal, entropy_3d_array_normal),
                                        TARGET_FATIGUE : (epochSeries_fatigue, frequencyFeatureDf_fatigue, entropy_3d_array_fatigue),
                                        TARGET_UNLABELED : (epochSeries_unlabeled, frequencyFeatureDf_unlabeled, entropy_3d_array_unlabeled)}
            else:
                print ("********* Skipping dir '{}' *********".format(subjectDir))

    return datasetDict



def createXyFromDataSeries(dataSeries : pd.Series, target : int) -> (np.ndarray, np.ndarray):
    ''' Create X and y for machine learning
    
    @param pd.Series dataSeries: Should be a series of dataframes
    
    X should look like this [samples, timesteps, features]
        epochs: the epoch
        samples: Number of samples in the epoch - E.g. if the epoch contains 200 values then the timestep should contain 200 values
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

def createXyFromFrequencyDf(freqDf : pd.DataFrame, target : int) -> (np.ndarray, np.ndarray):
    ''' Create X and y for machine learning from the frequency df
    
    @param pd.DataFrame freqDf: A dataframe containing all the features from the frequency level
    @param int target: The target of the dataframe, e.g. 0 or 1 which could correspond to fatigue or normal
    
    X should look like this [samples, timesteps, features]
        samples: The epoch
        timesteps: Only 1 timestep in this case - E.g. if the epoch contains 200 values then the timestep should contain 200 values
        features: The actual value
    
    y should look tlike this [classIds] 
        classIds: The label for the sample of the X Data
    '''
    
    X = freqDf.to_numpy() # returns a 2D np.array! E.g. 60 epochs with 1200 features
    
    # Make the array 3D Array -to e.g. 60 epochs, 1 timestep and 1200 features
    X = X.reshape(X.shape[0], 1, X.shape[1])
    
    y = [target] * X.shape[0]
    y = np.array(y)
    
    return X, y

def createXyFromEntropyArray(entropyArray : np.ndarray, target : int) -> (np.ndarray, np.ndarray):
    ''' Create X and y for the entropy Features
    
    However the entropy array is already good, we just have to make the y array
    '''
    # X is already good
    X = entropyArray
    
    # Create a y array with the amount of epochs
    y = [target] * X.shape[0]
    y = np.array(y)
    
    return X, y

def createMachineLearningDataset(eegDataset : Dict, targetLabelDict : Dict) -> (((np.array, np.array)),(np.array, np.array), (np.array, np.array)):
    ''' This functions creates from a loaded pickel dataset X and y data for the eeg data and frequency feature data
    
    The functions returns X_eegData, y_eegData and X_frequncyFeatures, y_frequencyFeatures
    Both X's are 3D numpy arrays, where:
        1D: epoch
        2D: samples - number of samples in the epoch
        3D: features - the actual value
    '''
    X_eeg_series = None
    y_eeg_series = None
    
    X_frequency_features = None
    y_frequency_features = None
    
    X_entropy_features = None
    y_entropy_features = None
    
    for subject in eegDataset:
        for target in targetLabelDict:
            
            try:
                print("Processing Subject {} - Target: {} ...".format(subject, target))
                
                # Create X,y for the eeg series
                tempX_eeg, tempy_eeg = createXyFromDataSeries(dataSeries = eegDataset[subject][target][0],
                                                              target = targetLabelDict[target])
                
                try:
                    X_eeg_series = np.concatenate((X_eeg_series, tempX_eeg))
                    y_eeg_series = np.concatenate((y_eeg_series, tempy_eeg))
                except ValueError: # happens the first, when the init value is none
                    X_eeg_series = tempX_eeg
                    y_eeg_series = tempy_eeg
                
                # Create X,y for the frequency features
                tempX_freq, tempy_freq = createXyFromFrequencyDf(freqDf = eegDataset[subject][target][1],
                                                                 target = targetLabelDict[target])
                
                try:
                    X_frequency_features = np.concatenate((X_frequency_features, tempX_freq))
                    y_frequency_features = np.concatenate((y_frequency_features, tempy_freq))
                except ValueError: # happens the first, when the init value is none
                    X_frequency_features = tempX_freq
                    y_frequency_features = tempy_freq
                    
                
                # Create X,y for the entropy features
                tempX_entropy, tempy_entropy = createXyFromEntropyArray(entropyArray = eegDataset[subject][target][2],
                                                                        target = targetLabelDict[target])
                
                try:
                    X_entropy_features = np.concatenate((X_entropy_features, tempX_entropy))
                    y_entropy_features = np.concatenate((y_entropy_features, tempy_entropy))
                except ValueError: # happens the first, when the init value is none
                    X_entropy_features = tempX_entropy
                    y_entropy_features = tempy_entropy
                    
            
            except TypeError:
                print("Skipping Target: {}".format(target))
    
    print("Done!")
    return ((X_eeg_series, y_eeg_series), (X_frequency_features, y_frequency_features), (X_entropy_features, y_entropy_features))

def createAndSafeMlDataset(eegDataset : Dict[str, Dict[str ,Tuple[pd.Series, pd.DataFrame]]], targetLabelDict : Dict,
                           dirPath : str) ->  ((np.array, np.array), (np.array, np.array)):
    
    if not os.path.isdir(dirPath):
        raise Exception("The given directory path is not valid! Given path: {}".format(dirPath))
    
    print("Creating Machine Learning Dataset!")
    eegData, freqData, entropyData = createMachineLearningDataset(eegDataset, targetLabelDict)

    
    print("\nSaving Machine Learning Dataset into this directory: {}".format(dirPath))
    np.save(os.path.join(dirPath, "X_eegData.npy"), eegData[0]) # X_eegData
    np.save(os.path.join(dirPath, "y_eegData.npy"), eegData[1]) # y_eegData

    np.save(os.path.join(dirPath, "X_frequencyData.npy"), freqData[0]) # X_frequencyData
    np.save(os.path.join(dirPath, "y_frequencyData.npy"), freqData[1]) # y_frequencyData

    np.save(os.path.join(dirPath, "X_entropyData.npy"), entropyData[0]) # X_frequencyData
    np.save(os.path.join(dirPath, "y_entropyData.npy"), entropyData[1]) # y_frequencyData
    
    # Save target labels
    saveDictToFile(targetLabelDict, filepath=os.path.join(dirPath,'target_labels.txt'))
    
    return eegData, freqData, entropyData

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



def loadOnlineEEGdata(dirPath="D:/Masterthesis/EEG_Data/eeg_data_online", splitData=True, test_size=0.3, shuffle=False, ) -> ((), ()):
    
    if not os.path.isdir(dirPath):
        raise Exception("Could not find '{}' - update the 'dirPath' param and say me where the directory of the eeg data is".format(dirPath))

    print("Loading Online EEG Data from {} ...".format(dirPath))
    
    # load array
    X_eeg = np.load(os.path.join(dirPath, 'X_eegData.npy'))
    y_eeg = np.load(os.path.join(dirPath, 'y_eegData.npy'))
    
    X_freq = np.load(os.path.join(dirPath, 'X_frequencyData.npy'), allow_pickle=True) # not sure, why I need this
    y_freq = np.load(os.path.join(dirPath, 'y_frequencyData.npy'))

    X_entropy = np.load(os.path.join(dirPath, 'X_entropyData.npy'))
    y_entropy = np.load(os.path.join(dirPath, 'y_entropyData.npy'))
    
    # load target labels
    
    # load feature names
    
    if splitData:
        # Split dataset into training set and test set
        # EEG Data
        X_eeg_train, X_eeg_test, y_eeg_train, y_eeg_test = train_test_split(X_eeg, y_eeg, test_size=test_size, shuffle=shuffle) # 70% training and 30% test    
        #y_eeg_train = to_categorical(y_eeg_train)
        #y_eeg_test = to_categorical(y_eeg_test)
        eegData = (X_eeg_train, y_eeg_train, X_eeg_test, y_eeg_test)
        print("EEG Data Shape:")
        print(X_eeg_train.shape, y_eeg_train.shape, X_eeg_test.shape, y_eeg_test.shape)
        
        # Frequency Data
        X_freq_train, X_freq_test, y_freq_train, y_freq_test = train_test_split(X_freq, y_freq, test_size=test_size, shuffle=shuffle) # 70% training and 30% test    
        #y_freq_train = to_categorical(y_freq_train)
        #y_freq_test = to_categorical(y_freq_test)
        freqData = (X_freq_train, y_freq_train, X_freq_test, y_freq_test)
        print("Freq Data Shape:")
        print(X_freq_train.shape, y_freq_train.shape, X_freq_test.shape, y_freq_test.shape)


        # Entropy Data
        X_entropy_train, X_entropy_test, y_entropy_train, y_entropy_test = train_test_split(X_entropy, y_entropy, test_size=test_size, shuffle=shuffle) # 70% training and 30% test
        entropyData = (X_entropy_train, y_entropy_train, X_entropy_test, y_entropy_test)
        print("Entropy Data Shape:")
        print(X_entropy_train.shape, y_entropy_train.shape, X_entropy_test.shape, y_entropy_test.shape)

    else:
        print("Data does not get splitted into train and test!")
        
        print("EEG Data Shape:")
        print(X_eeg.shape, y_eeg.shape)
        
        print("Freq Data Shape:")
        print(X_freq.shape, y_freq.shape)

        print("Entropy Data Shape:")
        print(X_entropy.shape, y_entropy.shape)

        eegData = (X_eeg, y_eeg)
        freqData = (X_freq, y_freq)
        entropyData = (X_entropy, y_freq)
    
    return (eegData, freqData, entropyData)

