#!/usr/bin/env python
'''
This file containts dicts, lists or functions related to data. 

This should have the advance, if you work with multiple notebooks you can import the data from here and don't have to create them again and again
'''

from consts import DEVICES_MUSE_LSL, DEVICES_OPEN_BCI, DEVICES_NEUROSCAN

testSubjectDict = { 1 : {"Device" : DEVICES_OPEN_BCI,
                     "awakeCsvPath" : "D:/OneDrive - bwedu/Masterthesis/Experiments+Data/Fahren+AimLab/2020_03_05_Propand_1/openBci_record-[2020.03.05-12.27.35]_raw_awake_aimlab.csv",
                     "unlabeledCsvPath" : "D:/OneDrive - bwedu/Masterthesis/Experiments+Data/Fahren+AimLab/2020_03_05_Propand_1/openBci_record-[2020.03.05-12.34.34]_raw_driving_unlabled.csv"},
                
               2 : {"Device" : DEVICES_MUSE_LSL,
                     "awakeCsvPath" : "",
                     "unlabeledCsvPath" : "D:/Masterthesis/Apps/eeg-notebooks/data/visual/P300/subject1/session1/data_2017-02-04-15_45_13.csv"}
                }


onlineEegDataDict = { 1 : {"Device" : DEVICES_NEUROSCAN,
                           "normalCsvPath" : "D:/Masterthesis/EEG_Data/eeg_data_online/1/Normal_state.csv",
                           "fatigueCsvPath" : "D:/Masterthesis/EEG_Data/eeg_data_online/1/Fatigue_state.csv"},
                           
                      2 : {"Device" : DEVICES_NEUROSCAN,
                           "normalCsvPath" : "D:/Masterthesis/EEG_Data/eeg_data_online/2/Normal_state.csv",
                           "fatigueCsvPath" : "D:/Masterthesis/EEG_Data/eeg_data_online/2/Fatigue_state.csv"}}