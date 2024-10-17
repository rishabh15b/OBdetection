import subprocess
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
import yaml
from yaml import SafeLoader
from pathlib import Path
from src.constants import Constant
from src.logger import get_logger
from src.error import Errors

LOGGER_ML = get_logger("Data Preprocessing & Training Logs")

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

with open(ROOT / Constant.CONFIG.value) as f:
    slice_config = yaml.load(f, Loader=SafeLoader)

with open(ROOT / Constant.TEST_CONFIG.value) as f:
    slice_config_test = yaml.load(f, Loader=SafeLoader)

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

images_folder = ROOT / slice_config['data_preparation']['processed_images_dir']
masks_folder = ROOT / slice_config['data_preparation']['processed_masks_dir']

data_preprocessing_path = ROOT / slice_config['data_preparation']['script_path']
data_preprocessing_path = data_preprocessing_path.as_posix()

training_path = ROOT / slice_config['train']['script_path']
training_path = training_path.as_posix()

inference_path = ROOT / slice_config_test['test']['script_path']
inference_path = inference_path.as_posix()

LOGGER_ML.info("Subprocess 1: DATA PREPROCESSING STARTED !")

choice = input("Enter 'train' to run training or 'test' to run testing: ")
    
if choice.lower() == "train":
    try:
        # if not os.path.exists(images_folder):
        subprocess.run(["python", data_preprocessing_path], check=True)
        # else:
            # LOGGER_ML.info("Data Preparation already done.")
    except:
        LOGGER_ML.error(Errors.code2.value, exc_info=True)

    LOGGER_ML.info("Subprocess 2: Training STARTED !")

    try:
        subprocess.run(["python", training_path], check=True)
    except:
        LOGGER_ML.error(Errors.code3.value, exc_info=True)

elif choice.lower() == "test":
    LOGGER_ML.info("Subprocess 3: Inference STARTED !")

    try:
        subprocess.run(["python", inference_path], check=True)
    except:
        LOGGER_ML.error(Errors.code4.value, exc_info=True)

else:
    LOGGER_ML.error(Errors.code5.value, exc_info=True)
