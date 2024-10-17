import os
from pathlib import Path
import yaml
import sys
from yaml import SafeLoader
import cv2
import json
import glob
import numpy as np
from logger import get_logger
from constants import Constant
from error import Errors
import warnings

warnings.filterwarnings("ignore")

LOGGER_ML = get_logger("Data Preparation Logs")

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

with open(ROOT / Constant.CONFIG.value) as f:
    slice_config = yaml.load(f, Loader=SafeLoader)

#create blank mask with image sizes
def create_binary_masks(im, shape_dicts):
    
    blank = np.zeros(shape=(im.shape[0], im.shape[1]), dtype=np.float32)
    for shape in shape_dicts:
        points = np.array(shape['points'], dtype=np.int32)
        cv2.fillPoly(blank, [points], 255)
    return blank

#get annotated points
def get_poly(ann_path):
    
    with open(ann_path) as handle:
        data = json.load(handle)
    shape_dicts = data['shapes']
    
    return shape_dicts

def create_mask_from_polygon(original_image, polygon_points_x, polygon_points_y):
    mask = np.zeros_like(original_image[:, :, 0], dtype=np.uint8)

    points = np.array(list(zip(polygon_points_x, polygon_points_y)), np.int32)
    points = points.reshape((-1, 1, 2))

    cv2.fillPoly(mask, [points], color=255)

    return mask


