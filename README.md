# Fatigue detection from driving persons with EEG - Masterthesis by Raphael Eißler

## Check the Wiki fore more Infos!!
- [First time setup](https://github.com/mudo121/thesis_eeg/wiki/First-time-Setup)
- [Improvements](https://github.com/mudo121/thesis_eeg/wiki/Possible-Improvements)
- [Bugs / Problems](https://github.com/mudo121/thesis_eeg/wiki/Known-Bugs-or-Issues)
- [Jupyter Basics for this Repo](https://github.com/mudo121/thesis_eeg/wiki/Jupyter-Notebook-Basics-for-using-existing-Functions)
- [EEG Data](https://github.com/mudo121/thesis_eeg/wiki/EEG-Data)
- [Driving Simulator](https://github.com/mudo121/thesis_eeg/wiki/Driving-Simulation)


# Description of the Repo
### 'Code' Folder
This folder contains everything related to the own programmed Pipeline. It contains of transformers, utility functions, plot functions, measering functions and functions to load/save/create Data.

### 'machine_learning_notebooks' Folder
This folder contains multiple jupyter notebooks for various created classifiers. 
 - **compare_classificators.ipynb** Creates various classifiers and creates a ROC Curve and compares it with each classifiers
 - **evaluate_models_detailed.ipynb** Evaluates multiple classifiers with a confusion matrix, accuracy, F1 Score, Recall and Precision and compares them with the different datasets.
 - **test_classifiers.ipynb** Here a grid search for multiple classifiers can be done.
 - **pauls_work.ipynb** (Neural Network Folder) Here multiple neural networks have been craeted and can be evaluated. (Done with the Help of Paul)
 - The other Notebooks are classifier specific.
 
### 'captureData' Folder
Contains the notebooks, which have been used for the experiments to capture the video camera and the Muse S EEG Data.

### 'Config' Folder
This Folder contains yaml config files for each used System. The config is partly explained in the code but also explained in the Thesis.
**TODO** - Write down each point of the config file

### 'data_processing' Folder
This folder contains notebooks to extract/process the raw EEG data from the experiments. There the 5 minutes of fatigue from the 40 minute drive will be extracted. Also the 5 minutes of awake will be extracted from the reaction game

### 'driving_simulator' Folder
This folder contains the scenario settings and files for the used driving simulator with OpenDS 4.5. It also contains a notebook to create waypoints for the auto "bots" to drive around.

### 'images' folder
This is just a temporary folder to store created images from the notebooks.

### 'openVibeScenarios' Folder
This fodler contains the created scenarios for openVibe. 

## 'old_notebooks' Folder
Contains a bunch of notebooks, which have been created but got not really used in the end.

### Repo Files Overview
- **processRawDataset.ipynb** - Shows the functions on how to process a raw .csv file into processed Files (pickeled) and machine learning ready files (.npy) and how they can get loaded.
- **thesis_diagrmas.ipynb** - A notebook which creates nearly all images needed for the Thesis.
- **experiment_protocol.odt** - Contains the used experiment protocol.
- **diagrams.pptx** - Contains multiple diagrams created for the thesis.

