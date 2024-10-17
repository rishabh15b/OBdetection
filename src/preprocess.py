import os
from pathlib import Path
import yaml
from yaml import SafeLoader
import sys
import json
import numpy as np
from glob import glob
import cv2
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils
from utils.helper import create_slices, arrange_samples
from utils.json_to_png import create_mask_from_polygon
from logger import get_logger
from constants import Constant
from error import Errors
import warnings

warnings.filterwarnings("ignore")

LOGGER_ML = get_logger("Data Preparation Logs")

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

with open(ROOT / Constant.CONFIG.value) as f:
    slice_config = yaml.load(f, Loader=SafeLoader)

# def convert_json_to_mask(image_folder_path, annotation_folder, masks_folder_path):
#     image_list = sorted(glob.glob(f'{image_folder_path}\*.JPG'))
#     ann_list = sorted(glob.glob(f'{annotation_folder}\*.json'))

#     #iterate every image and its json file to create binary mask
#     for im_fn, ann_fn in zip(image_list, ann_list):


#         file_path = os.path.join(masks_folder_path, os.path.basename(im_fn))
#         if os.path.isfile(file_path):
#             pass
#         else:
#             image = cv2.imread(im_fn)
#             im = cv2.imread(im_fn, 0)
#             shape_dicts = get_poly(ann_fn)
#             im_binary = create_binary_masks(im, shape_dicts)
        
#             #extract the name of image file
#             filename = im_fn.split('.JPG')[-2].split('/')[-1] + '.JPG'
    
#             cv2.imwrite(os.path.join(str(masks_folder_path), filename), im_binary)
    

def create_mask_from_json(image_folder, json_folder, masks_folder):
    
    json_list = sorted(glob(f'{json_folder}/*.json'))

    for json_path in json_list:
        filename, _ = os.path.splitext(os.path.basename(json_path))
        
        # Find the corresponding image with any extension
        image_list = glob(os.path.join(image_folder, f'{filename}.*'))
        if not image_list:
            continue  # Skip JSON files without corresponding images

        image_path = image_list[0]
        _, ext = os.path.splitext(image_path)

        original_image = cv2.imread(image_path)
        with open(json_path, 'r') as json_file:
            try:
                json_data = json.load(json_file)
            except json.JSONDecodeError:
                print(f"Error decoding JSON file: {json_path}")
                continue

        individual_masks = []

        if filename + ext in json_data and "regions" in json_data[filename + ext]:
          
            for region in json_data[filename + ext]["regions"]:
                polygon_points_x = region["shape_attributes"]["all_points_x"]
                polygon_points_y = region["shape_attributes"]["all_points_y"]

                # print(f"Processing {filename}: {polygon_points_x}, {polygon_points_y}")
                
                mask = create_mask_from_polygon(original_image, polygon_points_x, polygon_points_y)
                individual_masks.append(mask)
        elif "shapes" in json_data:
            shape_dicts = json_data["shapes"]
            for shape_dict in shape_dicts:
                label = shape_dict["label"]
                points = shape_dict["points"]
                
                mask = create_mask_from_polygon(original_image, *zip(*points))
                individual_masks.append(mask)

        final_mask = np.zeros_like(original_image[:, :, 0], dtype=np.uint8)
        for individual_mask in individual_masks:
            final_mask = cv2.bitwise_or(final_mask, individual_mask)

        cv2.imwrite(os.path.join(masks_folder, f'{filename}.JPG'), final_mask)  # Save as PNG to handle different image formats

def preprocess_data():
    create_slices(raw_image_folder_path, patched_images, num_cores)
    create_slices(raw_masks_folder_path, patched_masks, num_cores)

    arrange_samples(patched_masks, patched_images, processed_masks, processed_images)

if __name__ == "__main__":

    ## Preprocess_Data
    num_cores = os.cpu_count() - 1

    raw_image_folder_path = ROOT / slice_config['data_preparation']['rAW_image_folder_dir']
    annotation_folder_path = slice_config['data_preparation']['annotation_dir']
    
    raw_masks_folder_path = ROOT / slice_config['data_preparation']['rAW_masks_folder_dir']
    os.makedirs(raw_masks_folder_path, exist_ok=True)

    # convert_json_to_mask(raw_image_folder_path, annotation_folder_path, raw_masks_folder_path)

    create_mask_from_json(raw_image_folder_path, annotation_folder_path, raw_masks_folder_path)

    patched_images = ROOT / slice_config['data_preparation']['patched_images_dir']
    patched_masks = ROOT / slice_config['data_preparation']['patched_masks_dir']

    os.makedirs(patched_images, exist_ok=True)
    os.makedirs(patched_masks, exist_ok=True)

    processed_images = ROOT / slice_config['data_preparation']['processed_images_dir']
    processed_masks = ROOT / slice_config['data_preparation']['processed_masks_dir']
    
    os.makedirs(processed_images, exist_ok=True)
    os.makedirs(processed_masks, exist_ok=True)
    
    if os.path.exists(processed_masks) and os.path.exists(processed_images) and os.listdir(processed_masks) and os.listdir(processed_images):
        print('Folder exits already')
    else:
        preprocess_data()