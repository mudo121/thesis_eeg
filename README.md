# Fatigue detection from driving persons with EEG - Masterthesis by Raphael Eißler

## Check the Wiki fore more Infos!!
- [First time setup]()
- [Improvements](https://github.com/mudo121/thesis_eeg/wiki/Possible-Improvements)
- [Bugs / Problems](https://github.com/mudo121/thesis_eeg/wiki/Known-Bugs-or-Issues)
- [Jupyter Basics for this Repo](https://github.com/mudo121/thesis_eeg/wiki/Jupyter-Notebook-Basics-for-using-existing-Functions)
- [EEG Data](https://github.com/mudo121/thesis_eeg/wiki/EEG-Data)



# Description of the Repo
### Code Folder
This folder contains everything related to the own programmed Pipeline. It contains of transformers, utility functions, plot functions, measering functions and functions to load/save/create Data.

### Config Folder
This Folder contains yaml config files for each used System.
**TODO** - Write down each point of the config file

### Generated Data Folder
(Outdated) - Should be removed. The generated Data should be saved outside the Git because the Data can get too big.

### Jupyter Notebook Overview
- **main_notebook** - (currently outdated) - Includes some developed Code, which is now being used
- **processRawDataset** - Shows the functions on how to process a raw .csv file into processed Files (pickeled) and machine learning ready files (.npy)
- **test_online_eeg_data** - Notebook which has processed the available [online EEG Data](https://figshare.com/articles/The_original_EEG_data_for_driver_fatigue_detection/5202739). This Data was in a different format and needed to be downsampeled.
- **test_pipeline** - Notebook which calls parts of the used pipeline indepently, for testing purposes
- **testMuse** - Notebook to test/play with Muse Data (a bit outdated and hasn't been tested with the pipeline yet)
- Machine Learning Notebooks - Contains notebooks which are using machine learning ready data
  - **lstm** - Long-Short Term Memory Notebook (Keras) - Tested with online EEG Data
  - **knn** - K-near Neihbours Notebook (Scikit Learn) - Tested with online EEG Data
  - **svm** - Support Vector Machine Notebook (Scikit Learn) - Tested with online EEG Data
