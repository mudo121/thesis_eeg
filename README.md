# Fatigue detection from driving persons with EEG - Masterthesis by Raphael Eißler

## Getting Started
- Prepare Workspace
  - Install [Anaconda](https://www.anaconda.com/products/individual)
  - Create enviorment from file: `conda env create -f environment.yml` (use this [environment.yaml](environment.yml)) (Does'nt work for now...)
  - Install the conda enviorment into jupyter: `python -m ipykernel install --user --name myenv --display-name "Python (myenv)"` (more infos [here](https://ipython.readthedocs.io/en/stable/install/kernel_install.html))
- Clone this Repo on to your pc - (Optionally use the GitHub Software)
- Download the EEG Data from [here](https://1drv.ms/u/s!AuIx_mQRobFA1g8FMEXZBfbdnIgg?e=nMWSxN)
- Start Coding
  - Open a anconda powershell
  - Activate your enviorment: `conda activate yourEnv`
  - Change Directory where you want to start jupyter lab: `cd /path/to/dir`
  - Start Jupyter Lab: `jupyter lab`
  - Create your own Notebook for Coding
 
## Possible Code Improvements
- Outlier detection could be improved. So far it is only a simple threshold function

## Current Bugs
- The 'delete faulty epochs' in the function `prepare_Signal` is not working currently. Somehow this produces too many NaN Values afterwards. Most of the frequency features are getting NaN's even though I replace all NaN's with zeroes...


## Notebook Basics
### Local Imports
With this code you can locally import other functions
```
# to enable local imports
module_path = os.path.abspath('../code')
print(module_path)
if module_path not in sys.path:
    sys.path.append(module_path)
```

### Load the online EEG Data
With this function you can load the online EEG Data for Machine Learning
```
eegData, freqData = loadOnlineEEGdata(splitData=False)
eegX, eegy = eegData
freqX, freqy = freqData
```
or
```
eegData, freqData = loadOnlineEEGdata(splitData=True)
eegX_train, eegy_train, eegX_test, eegy_test = eegData
freqX_train, freqy_train, freqX_test, freqy_test = freqData
```

This functions has 3 Parameters:
- splitData (bool): True -> Splits the Data already in Train and Test Data
- test_size (float): 0.3 -> If splitData is true then 30% of the data will be test data
- shuffle (bool): If True, it will shuffle the data


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
