import os
from pathlib import Path
import yaml
from yaml import SafeLoader
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
import numpy as np
import torch
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math
from PIL import Image
import segmentation_models_pytorch as smp
from sahi.slicing import slice_image
import random
import shutil
from concurrent.futures import ThreadPoolExecutor
import cv2
from constants import Constant
from logger import get_logger
from error import Errors

import warnings
warnings.filterwarnings("ignore")

LOGGER_ML = get_logger("Data Preprocessing & Training Logs")

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

with open(ROOT / Constant.CONFIG.value) as f:
    slice_config = yaml.load(f, Loader=SafeLoader)

with open(ROOT / Constant.TEST_CONFIG.value) as f:
    slice_config_test = yaml.load(f, Loader=SafeLoader)

classes = slice_config['train']['classes']
font_path = slice_config_test['test']['font_path']
image_font = slice_config_test['test']['image_font']

def model_definition(Arc = slice_config['train']['architecture'],
    encoder = slice_config['train']['encoder'],
    encoder_weights = slice_config['train']['encoder_weights'],
    activation = 'sigmoid'):

    """
    Creates a new instance of Unet model from the Segmentation Models PyTorch library.

    Args:
        Arc (str): Not used in the function.
        encoder (str): The name of the encoder architecture to use for the Unet model.
        encoder_weights (str): The name of the encoder weights to use for the Unet model.
        activation (str): The name of the activation function to use for the Unet model. Default is 'sigmoid'.

    Returns:
        model (smp.Unet): The newly created Unet model.
        encoder (str): The name of the encoder architecture used for the model.
        encoder_weights (str): The name of the encoder weights used for the model.
        Arc (str): The value of the Arc parameter, which is not used in the function.
    """
    
    model = smp.Unet(
        encoder_name=encoder, 
        encoder_weights=encoder_weights, 
        classes= len(classes), 
        activation=activation,
    )
    return model, encoder, encoder_weights, Arc

def exp_size(image, pat_sz):
    """
    Expands the size of an image to a multiple of a specified patch size.

    Args:
        image (numpy.ndarray): The input image as a numpy array.
        pat_sz (int): The desired patch size.

    Returns:
        list: A list containing the expanded width and height of the image.
    """
    exp = []
    im_height,im_width = image.shape[1],image.shape[0]
    exp_height = int(math.ceil(float(im_height) / pat_sz)) * pat_sz
    exp_width = int(math.ceil(float(im_width) / pat_sz)) * pat_sz
    exp.append(exp_width)
    exp.append(exp_height)
    return exp

def add_pad(image, x):
    """
    Adds padding to an image to make it a multiple of a specified patch size.

    Args:
        image (numpy.ndarray): The input image as a numpy array.
        x (list): The desired width and height of the padded image.

    Returns:
        numpy.ndarray: The padded image as a numpy array.
    """
    pixels_to_add_h = x[0] - image.shape[0]
    pixels_to_add_w = x[1] - image.shape[1]
    pad_h = np.zeros((pixels_to_add_h,image.shape[1], 3))
    result_h = np.vstack((image,pad_h))
    pad_w = np.zeros((result_h.shape[0], pixels_to_add_w,3))
    result = np.hstack((result_h, pad_w))
    return result 

def add_pad_one(image, x):
    """
    Adds padding to an image to make it a multiple of a specified patch size, while preserving the original aspect ratio.

    Args:
        image (numpy.ndarray): The input image as a numpy array.
        x (list): The desired width and height of the padded image.

    Returns:
        numpy.ndarray: The padded image as a numpy array.
    """
    pixels_to_add_h = x[0] - image.shape[0]
    pixels_to_add_w = x[1] - image.shape[1]
    pad_h = np.zeros((pixels_to_add_h,image.shape[1]))
    result_h = np.vstack((image,pad_h))
    pad_w = np.zeros((result_h.shape[0], pixels_to_add_w))
    result = np.hstack((result_h, pad_w))
    return result

def sulphur_preds(sample, model1):
    """
    Makes a prediction for a sulphur mask using a trained Unet model.

    Args:
        sample (numpy.ndarray): The input image as a numpy array.
        model1 (smp.Unet): A trained Unet model.

    Returns:
        numpy.ndarray: The predicted sulphur mask as a numpy array.
    """
    DEVICE = 'cuda'
    x_tensor = torch.from_numpy(sample).to(DEVICE).unsqueeze(0)
    pr_mask = model1.predict(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())
    pr_mask_scaled = (pr_mask * 255).astype(np.uint8)
    return pr_mask_scaled

def move_files(source_dir, dest_dir, files):
    """
    Moves a list of files from a source directory to a destination directory.

    Args:
        source_dir (str): The source directory.
        dest_dir (str): The destination directory.
        files (list): A list of file names to move.
    """
    os.makedirs(dest_dir, exist_ok=True)
    for file in files:
        source_path = os.path.join(source_dir, file)
        dest_path = os.path.join(dest_dir, file)
        os.rename(source_path, dest_path)

classes_test =  slice_config_test['test']['classes']

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def merge_overlaped_boxes(sorted_bbox_list):
    """Merges overlapping bounding boxes in a sorted list of bounding boxes.
    
    Each bounding box is represented as a list of four integers: [x1, y1, x2, y2],
    where (x1, y1) and (x2, y2) are the coordinates of the top-left and
    bottom-right corners of the box, respectively. The list of bounding boxes
    is assumed to be sorted in increasing order of y1.
    
    This function modifies the input list in-place and returns the modified list.
    
    Args:
        sorted_bbox_list: A list of bounding boxes, sorted in increasing order of y1.
    
    Returns:
        The input list with overlapping bounding boxes merged.
    """
    i=0
    while i <len(sorted_bbox_list):
        sorted_bbox_list = sorted(sorted_bbox_list, key=lambda box: (box[1]))
        current_box = sorted_bbox_list[i]
        j=i+1
        while j < len(sorted_bbox_list):
            next_box= sorted_bbox_list[j]
            if is_overlap(current_box, next_box):
                current_box = merge_boxes(current_box, next_box)
                sorted_bbox_list.pop(j)
                sorted_bbox_list.pop(i)
                sorted_bbox_list.append(current_box)
                
                sorted_bbox_list = sorted(sorted_bbox_list, key=lambda box: (box[1]))
            
                i=0
                break
            j = j+1 
        i=i+1
            
    return sorted_bbox_list

def box_to_point(bbox):
    """
    Converts a bounding box to a list of points.

    Args:
        box (list): A list of four integers representing the coordinates of the bounding box.

    Returns:
        list: A list of points, where each point is represented as a list of two integers.

    """
    top_left = Point(bbox[0], bbox[1])
    bot_right = Point(bbox[2], bbox[3])
    return top_left, bot_right

def point_to_box(top_left, bot_right):
    return [int(top_left.x), int(top_left.y), int(bot_right.x), int(bot_right.y)]

def merge_boxes(bbox1, bbox2):
    box1_top_left, box1_bot_right = box_to_point(bbox1[0:4])
    box2_top_left, box2_bot_right = box_to_point(bbox2[0:4])
    merge_top_left, merge_bot_right = Point(0, 0), Point(0, 0)

    # score_list = [bbox1[4], bbox2[4]]
    # label_list = [bbox1[5], bbox2[5]]
    # idx = np.argmax(score_list)

    merge_top_left.x = min(box1_top_left.x, box2_top_left.x)
    merge_top_left.y = min(box1_top_left.y, box2_top_left.y)
    merge_bot_right.x = max(box1_bot_right.x, box2_bot_right.x)
    merge_bot_right.y = max(box1_bot_right.y, box2_bot_right.y)

    return [*point_to_box(merge_top_left, merge_bot_right)]


def check_overlap(p1, p2):
    return (p1.x <= p2.x <= p1.y) or (p2.x <= p1.x <= p2.y)


def is_overlap(box1, box2):
    box1_top_left, box1_bot_right = box_to_point(box1[0:4])
    box2_top_left, box2_bot_right = box_to_point(box2[0:4])

    box1_projection_x = Point(*sorted([box1_top_left.x, box1_bot_right.x]))
    box1_projection_y = Point(*sorted([box1_top_left.y, box1_bot_right.y]))
    box2_projection_x = Point(*sorted([box2_top_left.x, box2_bot_right.x]))
    box2_projection_y = Point(*sorted([box2_top_left.y, box2_bot_right.y]))

    if check_overlap(box1_projection_x, box2_projection_x) and check_overlap(box1_projection_y, box2_projection_y):
        return True
    return False

def box_building(mask, label, score=100):
    """
    Builds bounding boxes around connected components in a mask.

    Args:
        mask (numpy.ndarray): A binary mask.
        label (int): The label to use for the connected components.
        score (int): The score to assign to the bounding boxes.

    Returns:
        list: A list of bounding boxes.
    """
    
    bbox_list = []
    area_list = []
    # Combining overlapping masks
    new_mask = np.zeros(mask.shape)
    contours, _ = cv2.findContours(
        mask[:, :, 0].astype("uint8"), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    approx_contour = [
        cv2.approxPolyDP(c, 0.005 * cv2.arcLength(c, True), True) for c in contours
    ]
    # Extracting x and y coordinates from approx contours
    for cont in approx_contour:
        # Checking if the annotations are valid
        x_coord = cont[:, 0][:, 0].tolist()
        y_coord = cont[:, 0][:, 1].tolist()
        if (len(x_coord) < 3) | (len(y_coord) < 3):
            continue
        pts = np.array([[(x, y) for x, y in zip(x_coord, y_coord)]])
        cv2.fillPoly(new_mask, pts=pts, color=(255, 255, 255))

    # Building boxes
    contours, _ = cv2.findContours(
        new_mask[:, :, 0].astype("uint8"), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    new_mask = np.zeros(mask.shape)
    # Extracting x and y coordinates from approx contours
    for cont_original in contours:
        # Checking if the annotations are valid
        cont = cv2.approxPolyDP(
            cont_original, 0.01 * cv2.arcLength(cont_original, True), True
        )
        x_coord = cont[:, 0][:, 0].tolist()
        y_coord = cont[:, 0][:, 1].tolist()
        if (len(x_coord) < 3) | (len(y_coord) < 3):
            continue
        pts = np.array([[(x, y) for x, y in zip(x_coord, y_coord)]])
        bbox = [np.min(x_coord), np.min(y_coord), np.max(x_coord), np.max(y_coord)]
        if (bbox[2] - bbox[0] < 100) and (bbox[3] - bbox[1] < 100):
            continue

        area = cv2.contourArea(cont_original)

        if area > 7500:
            bbox_list.append(bbox)
            area_list.append(area)
    return bbox_list, area_list

def draw_bbox_on_img(img, bbox_list,area_list):
    res_list = []
    bbox_list = merge_overlaped_boxes(bbox_list)
    bbox_list = merge_overlaped_boxes(bbox_list)
    for i in range(len(bbox_list)):
        bbox =bbox_list[i]
        area = area_list[i]
        avg_score = 1
        class_name = slice_config_test['test']['classname']
        class_name_mod = slice_config_test['test']['classname']
        color_list = [item for sublist in classes_test["color"] for item in sublist]
        img = draw_bounding_box(
            img.astype(np.uint8),
            bbox,
            slice_config_test['test']['text_color_list'],
            slice_config_test['test']['text_font_size'],
            color_list,
            slice_config_test['test']['bbox_thickness'],
            class_name_mod,
            avg_score,
        )

        res_list.append(
            {
                "type": class_name,
                "score": avg_score,
                "bbox": bbox,
                "bboxCenter": [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
                "area": area,
            }
        )

    return img, res_list

def draw_bounding_box(
    img,
    box,
    text_color_list,
    text_font_size,
    bbox_color_list,
    bbox_thickness,
    id_2_class,
    scores,
):
    image = Image.fromarray(img)
    font = ImageFont.truetype(
        font_path, size=text_font_size
    )
    bold_font = ImageFont.truetype(image_font,size=text_font_size)

    format_score = "| " + str(int(scores * 100))+"%"
    concat_label = "{uc_name}".format(uc_name = id_2_class)

    xmin, ymin, xmax, ymax = box
    text_color = tuple(text_color_list)
    bbox_color = tuple(bbox_color_list)
    start_point = (xmin, ymin)
    end_point = (xmax, ymax)

    width = xmax - xmin
    height = ymax - ymin
    area = width * height

    if area < 10000:
        pass
    
    else:
        draw = ImageDraw.Draw(image)
        draw.rectangle((start_point, end_point), outline=bbox_color, width=bbox_thickness)

        text_width = draw.textlength(concat_label, font=bold_font)

        # Padding
        x_pad = 20
        y_pad = 20
        text_width =  text_width + x_pad
        text_height = 80

        draw.rectangle([(xmin, ymin - text_height), (xmin + text_width, ymin)], fill=bbox_color, width=bbox_thickness)
        draw.rectangle([(xmin + text_width, ymin - text_height), (xmin + text_width + 130, ymin)], fill=bbox_color, width=bbox_thickness)

        text_x = xmin + x_pad//2
        text_y = ymin - 0.8*text_height

        draw.text((text_x, text_y),concat_label ,font=bold_font, fill= text_color)

        text_x = xmin + text_width + x_pad//4
        text_y = ymin - 0.8*text_height
        # draw.text((text_x, text_y),format_score, font=font, fill= text_color)

    return np.array(image, dtype=np.uint8)

def has_specific_pixels(img_rgb, pixel_values):
    """
    Checks if an image contains any of the specified pixelmax - xmin
    height = ymax - ymin
    area = width * height

    values.

        Args:
            img_rgb (numpy.ndarray): An    if area < 10000:
            pass

        else:
            draw = ImageDraw RGB image.
            pixel_values (list): A list of pixel values to check for.

        Returns:
            bool.Draw(image)
            draw.rectangle((start_point, end_point), outline=bbox_: True if any of the specified pixel values are found in the image, False otherwise.
    """
    
    # Convert the list of pixel values to a NumPy array
    pixel_values_array = np.array(pixel_values, dtype=np.uint8)

    # Check if any pixel in the image matches any pixel value in the list
    for pixel_value in pixel_values_array:
        mask = np.all(img_rgb == pixel_value, axis=-1)
        if np.any(mask):
            return True

    # If no matching pixels are found
    return False

def resize_image(image, scaling_factor):
    resized_image = cv2.resize(
        image,
        (
            int(image.shape[1] * scaling_factor),
            int(image.shape[0] * scaling_factor),
        ),
    )
    return resized_image


## Preproces function

def preprocess_mask(mask):
    """
    Preprocesses a mask by conver draw.text((text_x, text_y),format_scoreting it to a binary mask and scaling the values.

    Args:
        mask (numpy.ndarray): A mask.

    Returns:
        numpy, font=.ndarray: The preprocessed mask.
    """
    mask = mask.astype(np.float32)
    mask[mask <= 150] = 0.0
    mask[mask > 150] = 1.0
    return mask

def find_zero_mask(mask):
    return set(np.unique(mask)) == {0.0}

def is_image_completely_black(image_path):
    """
    Checks if an image is completely black.

    Args:
        image_path (str): list of pixel values to check for.

    Returns:
        bool: True if any of the specified pixel values are The found in the image, False otherwise.
    """
    mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask_pr = preprocess_mask(mask)
    return find_zero_mask(mask_pr)

def create_sample_zero_nonzero_images_mix(mask_folder):
    """
    Creates a list of completely black and_values_array:
        mask = np.all(img_ non-black masks.

    Args:
        mask_folder (str): The folder containing the masks.

    Returns:
        tuple: A tuple containing two lists, one for completely black masks and one for non-black masks.
    """
    zero_images = []
    nonzero_images = []
    for mask_file in os.listdir(mask_folder):
        mask_img_path = os.path.join(mask_folder, mask_file)
        if is_image_completely_black(mask_img_path):
            zero_images.append(mask_img_path)
        else:
            nonzero_images.append(mask_img_path)
    return zero_images, nonzero_images

def create_equal_sample_of_zero_nonzero_images_from_list(zero_images, nonzero_images):
    random.shuffle(zero_images)
    random.shuffle(nonzero_images)

    if len(zero_images) > len(nonzero_images):
        zero_images = zero_images[:len(nonzero_images)]
    else:
        nonzero_images = nonzero_images[:len(zero_images)]
    return zero_images, nonzero_images

def copy_images(zero_images, nonzero_images, new_mask_folder, new_image_folder):

    mask_dir_name = os.path.dirname(zero_images[0])
    parent_dir_name = os.path.dirname(mask_dir_name)

    for zero_image in zero_images:
        basename_mask_file = os.path.basename(zero_image)
        corresponding_original_img_path = os.path.join(parent_dir_name,"patched_images", basename_mask_file)
        new_mask_file = os.path.join(new_mask_folder, basename_mask_file)
        new_image_file = os.path.join(new_image_folder, basename_mask_file)
        shutil.copyfile(zero_image, new_mask_file)
        shutil.copyfile(corresponding_original_img_path, new_image_file)
    
    for nonzero_image in nonzero_images:
        basename_mask_file = os.path.basename(nonzero_image)
        corresponding_original_img_path = os.path.join(parent_dir_name, "patched_images", basename_mask_file)
        new_mask_file = os.path.join(new_mask_folder, basename_mask_file)
        new_image_file = os.path.join(new_image_folder, basename_mask_file)
        shutil.copyfile(nonzero_image, new_mask_file)
        shutil.copyfile(corresponding_original_img_path, new_image_file)

def arrange_samples(mask_folder, original_folder, new_mask_folder, new_image_folder):

    all_zero_images, all_nonzero_images = create_sample_zero_nonzero_images_mix(mask_folder)
    zero_images, nonzero_images = create_equal_sample_of_zero_nonzero_images_from_list(all_zero_images, all_nonzero_images)
    copy_images(zero_images, nonzero_images, new_mask_folder, new_image_folder)

def create_slices(img_dir, slice_dir, num_cores):
    """
    Slices the images in the `img_dir` directory and saves the slices in the `slice_dir` directory.

    Args:
        img_dir (str or Path): The path to the directory containing the images to slice.
        slice_dir (str or Path): The path to the directory where the slices will be saved.
        num_cores (int): The number of cores to use for slicing the images.
        slice_height (int): The height of each slice. Default: 1024.
        slice_width (int): The width of each slice. Default: 1024.
        overlap_height_ratio (float): The ratio of the height overlap between slices. Default: 0.
        overlap_width_ratio (float): The ratio of the width overlap between slices. Default: 0.
        min_area_ratio (float): The minimum area ratio of the objects in the slices. Default: 0.2.

    """
    slice_height = 1024
    slice_width = 1024
    overlap_height_ratio = 0
    overlap_width_ratio = 0

    images_names = os.listdir(img_dir)

    if len(images_names) <= 0:
        LOGGER_ML.error(Errors.code1.value.format(img_dir), exc_info=True)
        # raise ValueError("No images found in {}".format(img_dir))
    
    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        LOGGER_ML.info("Slicing images in {}".format(slice_dir))

        for image_file in images_names:
            image_path = os.path.join(img_dir, image_file)
            executor.submit(slice_image, image=image_path, 
                            output_file_name=image_file.split('.')[0],
                            output_dir=slice_dir, slice_height=slice_height, 
                            slice_width=slice_width, overlap_height_ratio=overlap_height_ratio, 
                            overlap_width_ratio=overlap_width_ratio, 
                            min_area_ratio = 0.2)

def delete_files_in_directory(directory_path):
    """
    Deletes all files in the given directory.

    Args:
        directory_path (str or Path): The path to the directory.

    Raises:
        OSError: If there is an error accessing the directory.

    """
    try:
        files = os.listdir(directory_path)
        for file in files:
            file_path = os.path.join(directory_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    except OSError:
        pass

def create_experiment_folder():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    experiment_name = f"experiment_{timestamp}"
    experiment_path = os.path.join("experiments", experiment_name)

    os.makedirs(experiment_path)
    return experiment_path