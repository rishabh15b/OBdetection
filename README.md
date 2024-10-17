# Obdetection

## Description

This repository preprocess the dataset with stockpile images and performs training using UNET architecture, Efficientnetb7 as encoder to segment stockpile.

## Installation
1. Clone this repository 

```git clone https://github.com/rishabh15b/OBdetection.git```

Switch to main branch

```git checkout "main"```

2. Run :

```conda create --name <ENVNAME> python=3.9```

3. Activate the installed conda environment :

```conda activate <ENVNAME>```

4. Install the required packages by running these commands in sequence,

```pip install --no-cache-dir -r requirements.txt```

```pip install pkg_install/segmentation_models.pytorch-0.3.2.tar.gz```

## Usage
Before running the codes, 


1. Move the images to "data/raw_data/images" and masks to "data/raw_data/masks" path.
2. Move the blind images to this "data/blind_data" path.
3. Do change the CLASSNAME with respective classname in both the config files.

NOTE (1): The name of the images file and the names of masks file should match !!!

Example of folder structure must look like this :

## Folder Structure
    config
        - config_train.yaml
        - config_test.yaml

    data
        - blind_data
            
        - raw_data
            - images
            - masks
    
    experiments
        - models
            - plots

        - prediction

    fonts

    logs
        - app.log

    pkg_install

    src 
        utils
            - dataset.py
            - helper.py
            - plot.py
        - constants.py
        - error.py
        - inference.py
        - logger.py
        - preprocess.py
        - train.py

    app.py  
          
## Generation of data for training and results from the model
Run the following from root directory of the project,
It will ask user to input "train": to process the data and start training or "test": to get prediction on blind data.

```python app.py```

*** During training, users define experiment names in the config_train.yaml file, where the system creates folders named after these experiments in the "experiments" directory to store models and plots. 
&& In testing, users specify the experiment name in the config_test.yaml file, and predictions are saved in the corresponding experiment folder. Also the config_train.yaml and config_test.yaml for each experiments are moved to the respective folder.

## Authors and acknowledgment
This project is authored by Rishabh Balaiwar <rishabh.balaiwar@gmail.com>.