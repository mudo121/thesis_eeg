#!/usr/bin/env python
"""Justa file to load some data. Made for users, which dont want to import all the stuff"""

import os
import numpy as np
from sklearn.model_selection import train_test_split


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
        print("X train: {} --- y train: {} #### X test: {} ---- y test: {}".format(X_eeg_train.shape, y_eeg_train.shape, X_eeg_test.shape, y_eeg_test.shape))
        
        # Frequency Data
        X_freq_train, X_freq_test, y_freq_train, y_freq_test = train_test_split(X_freq, y_freq, test_size=test_size, shuffle=shuffle) # 70% training and 30% test    
        #y_freq_train = to_categorical(y_freq_train)
        #y_freq_test = to_categorical(y_freq_test)
        freqData = (X_freq_train, y_freq_train, X_freq_test, y_freq_test)
        print("Freq Data Shape:")
        print("X train: {} --- y train: {} #### X test: {} ---- y test: {}".format(X_freq_train.shape, y_freq_train.shape, X_freq_test.shape, y_freq_test.shape))


        # Entropy Data
        X_entropy_train, X_entropy_test, y_entropy_train, y_entropy_test = train_test_split(X_entropy, y_entropy, test_size=test_size, shuffle=shuffle) # 70% training and 30% test
        entropyData = (X_entropy_train, y_entropy_train, X_entropy_test, y_entropy_test)
        print("Entropy Data Shape:")
        print("X train: {} --- y train: {} #### X test: {} ---- y test: {}".format(X_entropy_train.shape, y_entropy_train.shape, X_entropy_test.shape, y_entropy_test.shape))

    else:
        print("Data does not get splitted into train and test!")
        
        print("EEG Data Shape:")
        print("X: {} --- y: {}".format(X_eeg.shape, y_eeg.shape))
        
        print("Freq Data Shape:")
        print("X: {} --- y: {}".format(X_freq.shape, y_freq.shape))

        print("Entropy Data Shape:")
        print("X: {} --- y: {}".format(X_entropy.shape, y_entropy.shape))

        eegData = (X_eeg, y_eeg)
        freqData = (X_freq, y_freq)
        entropyData = (X_entropy, y_entropy)
    
    return (eegData, freqData, entropyData)