import os
import cv2
import torch
import numpy as np
from pathlib import Path
import yaml
from yaml import SafeLoader
import sys
from utils.helper import has_specific_pixels, box_building, draw_bbox_on_img, resize_image
import albumentations as albu
from patchify import unpatchify
from datetime import datetime
from ast import literal_eval as make_tuple
from tqdm import tqdm 
from src.logger import get_logger
from src.constants import Constant
# from src.error import Errors
import warnings

warnings.filterwarnings("ignore")

LOGGER_ML = get_logger("Inference Logs")

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

with open(ROOT / Constant.TEST_CONFIG.value) as f:
    slice_config_test = yaml.load(f, Loader=SafeLoader)


device = slice_config_test['test']['device']
REDUCED_FLAG = slice_config_test['test']['reduced_flag']
scaling_factor = slice_config_test['test']['scaling_factor']
classes = slice_config_test['test']['classes']

patch_size = make_tuple(slice_config_test['test']['patch_size'])
stride = make_tuple(slice_config_test['test']['stride'])
score_thresh = float(slice_config_test['test']['score_thresh'])
mask_thresh = float(slice_config_test['test']['mask_thresh'])

def extract_patch_info(img, patch_size, stride):
    n1, n2, n3 = (
        img.shape[0] // patch_size[0],
        img.shape[1] // patch_size[1],
        img.shape[2] // patch_size[2],
    )
    return n1, n2, n3

def preprocess_image(img, patch_size, stride):
    n1, n2, n3 = extract_patch_info(
        img, patch_size, stride
    )

    img_tensor = torch.from_numpy(img)
    img_patches = (
        img_tensor.unfold(0, patch_size[0], stride[0])
        .unfold(1, patch_size[1], stride[1])
        .unfold(2, patch_size[2], stride[2])
    )
    img_patches = np.asarray(img_patches, dtype=np.uint8)
    return img_patches, n1, n2, n3

def save_image(image, output_path):
    cv2.imwrite(output_path, image)

def load_model():

    model_path = slice_config_test['test']['model_path'] + "/" + slice_config_test['test']['experiment_name'] + "/" + slice_config_test['test']['model_name']

    prediction_model = torch.load(model_path, device)
    
    LOGGER_ML.info("Model loaded successfully")

    return prediction_model

prediction_model = load_model()

def get_image_name(img_path):
    return os.path.splitext(os.path.basename(img_path))[0]

def load_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def predict_patches(
    img_patches, device, prediction_model, classes, score_thresh, mask_thresh
):
    pred_list = []
    reconstructed_img = np.zeros(img_patches.shape)
    stock_mask = np.zeros(img_patches.shape)

    score = 100

    pixel_values = [[24, 97, 111], [66, 102, 103], [42, 78, 78], [20, 51, 68], [51, 124, 126],
                    [21, 58, 71], [139, 155, 163], [29, 75, 92], [36, 66, 79], [36, 84, 96],
                    [28, 76, 84], [4, 44, 60], [135, 169, 196], [66, 103, 141], [92, 124, 160],
                    [83, 124, 156], [156, 183, 203], [109, 139, 171], [115, 148, 172], [72, 86, 59],
                    [143, 153, 139], [106, 118, 98], [163, 180, 163], [70, 67, 57],
                    [32,138,150], [39, 146, 159], [73,154,163], [43, 167, 172], [106, 109, 112]
                    ]
    threshold = 0.5
       
    for patch_i in range(img_patches.shape[0]):
        for patch_j in range(img_patches.shape[1]):
            patch = img_patches[patch_i][patch_j][0]

            ##check
            pixcel_check = has_specific_pixels(patch, pixel_values)
    
            # Check if the bluish percentage is above the threshold
            if pixcel_check :
                binary_mask = np.ones([patch.shape[0], patch.shape[1]], dtype=np.uint8)
                binary_mask = (binary_mask * 255).astype(np.uint8)
            
            else:
            
                patch_pil = albu.Normalize()(image=patch)
                patch_pil = patch_pil['image'].transpose(2, 0, 1)

                patch_pil = torch.from_numpy(patch_pil).to(device).unsqueeze(0)

                with torch.no_grad():
                    prediction = prediction_model(patch_pil)

                prediction_class_1 = prediction.squeeze().cpu().numpy()[0]
                binary_mask = (prediction_class_1 > threshold) * 255
                binary_mask = (binary_mask * 255).astype(np.uint8)
            
            reconstructed_img[patch_i][patch_j][0] = patch
            stock_mask[patch_i][patch_j][0] = np.stack((binary_mask, binary_mask, binary_mask), axis=-1)

    stock_mask[stock_mask > 255] = 255
    return pred_list, reconstructed_img, stock_mask, score

def post_process_results(
    reconstructed_img,
    stock_masks,
    pred_list,
    img_name,
    classes,
    patch_size,
    stride,
    n1,
    n2,
    n3,
    score,
):

    element = np.ones((5, 5), np.uint8)

    stock_mask = unpatchify(
        stock_masks, (patch_size[0] * n1, patch_size[1] * n2, patch_size[2] * n3)
    )

    reconstructed_img = unpatchify(
        reconstructed_img, (patch_size[0] * n1, patch_size[1] * n2, patch_size[2] * n3)
    )
    reconstructed_img = reconstructed_img.astype(np.uint8)
    stock_mask = stock_mask.astype(np.uint8)

    stock_mask_erode = cv2.dilate(stock_mask, element, iterations=10)
    stock_mask = cv2.erode(stock_mask_erode, element, iterations=5)
    
    stock_mask = stock_mask == (0, 0, 0)
    stock_mask = stock_mask * classes["color"][0]

    reconstructed_img = cv2.cvtColor(reconstructed_img, cv2.COLOR_RGB2BGR)

    reconstructed_img[reconstructed_img > 255] = 255

    # cv2.imwrite(f"masks/{img_name}_mask.JPG", stock_mask )
    # reconstructed_img, results = box_building(
    #     reconstructed_img, stock_mask, "stockpiles", score
    # )
    bbox_list, area_list  = box_building(
        stock_mask, "stockpiles", score
    )
    reconstructed_img, results = draw_bbox_on_img(reconstructed_img, bbox_list, area_list)

    pred_list = pred_list + results
    LOGGER_ML.info("box building done")
    recon_img_path = ""
    res_img_reduced_path = ""

    prediction_path = slice_config_test['test']['model_path'] + "/" + slice_config_test['test']['experiment_name'] + "/" + slice_config_test['test']['prediction_path']
    isExist = os.path.exists(prediction_path)
    if not isExist:
        os.makedirs(prediction_path)

    if len(pred_list) > 0:
        recon_img_path = f"{prediction_path}/{img_name}_processed_full_scaled.JPG"
        save_image(reconstructed_img, recon_img_path)
        # LOGGER_ML.info("Prediction exported")
        if REDUCED_FLAG == REDUCED_FLAG:
            # print("Inside : Constant.REDUCED_FLAG.value")
            res_img_reduced_path = f"{prediction_path}/{img_name}_processed_rescaled.JPG"

            LOGGER_ML.info("res_img_reduced_path:{}".format(res_img_reduced_path))
            bbox_img = np.array(reconstructed_img, dtype=np.uint8)
            bbox_img = cv2.cvtColor(bbox_img, cv2.COLOR_RGB2BGR)
            result_img_red = resize_image(bbox_img, scaling_factor)
            # print("Inside : result_img_red")
            cv2.imwrite(res_img_reduced_path, result_img_red[:,:,::-1])
            LOGGER_ML.info(
                "reduced image saved for img_name {} : {}".format(
                    img_name, res_img_reduced_path
                )
            )
            LOGGER_ML.info("Prediction exported")

        predicted_list_reshape = ["{l} {s} {x1} {y1} {x2} {y2}\n".format(l=p["type"], 
                                                                        s=p["score"], x1=p["bbox"][0], y1=p["bbox"][1], 
                                                                        x2=p["bbox"][2], y2=p["bbox"][3]
                                                                        ) 
                                for p in pred_list]
        predicted_filename_txt = f"{prediction_path}/{img_name}_pred.txt"
        # predicted_filename_txt = os.path.join(pred_out_dir, img_name+"_pred.txt")

        with open(predicted_filename_txt, "w") as outfile:
            outfile.writelines(predicted_list_reshape)
            LOGGER_ML.info("Prediction exported as _pred.txt : {}".format(predicted_filename_txt))
            
        return {
            "detection": pred_list,
            "pred_path": recon_img_path,
            "reduc_path": res_img_reduced_path if REDUCED_FLAG == True else None,
        }

    else:
        LOGGER_ML.error("Prediction is Empty")
        return {
            "detection": pred_list,
            "pred_path": recon_img_path,
            "reduc_path": res_img_reduced_path if REDUCED_FLAG == True else None,
        }
    
def model_pred(img_path):
    img_name = get_image_name(img_path)
    LOGGER_ML.info("AI Model Prediction started")
    start_time = datetime.now()
    pred_list = []

    # Load the image
    img = load_image(img_path)
    LOGGER_ML.info("img path (pred): {}".format(img_path))

    # Preprocessing
    img_patches, n1, n2, n3 = preprocess_image(img, patch_size, stride)

    LOGGER_ML.info("Prediction Started")

    # Model Prediction
    pred_list, reconstructed_img_patches, stock_mask_patches, score = predict_patches(
        img_patches, device, prediction_model, classes, score_thresh, mask_thresh
    )

    # Post Processing
    results = post_process_results(
        reconstructed_img_patches,
        stock_mask_patches,
        pred_list,
        img_name,
        classes,
        patch_size,
        stride,
        n1,
        n2,
        n3,
        score,
    )

    LOGGER_ML.info("Prediction exported")

    return results

if __name__ == "__main__":

    blind_images_data = slice_config_test['test']['blind_data_dir']

    for img_file in tqdm(os.listdir(blind_images_data)):
        # if img_file.startswith("DJI_"):
        if img_file.endswith('.JPG'):
            filename1 = img_file.split('.')[0] + '_processed_rescaled.JPG'
            filename2 = img_file.split('.')[0] + '_processed_full_scaled.JPG'
            file_path1 = os.path.join(blind_images_data, filename1 + '')
            file_path2 = os.path.join(blind_images_data, filename2 + '')
            if os.path.isfile(file_path1) or os.path.exists(file_path2):
                # print("File already there!!")
                pass
            else:
                LOGGER_ML.info(img_file)
                final_result = model_pred(os.path.join(blind_images_data, img_file))
    temp_path = slice_config_test['test']['model_path'] + "/" + slice_config_test['test']['experiment_name']
    with open(os.path.join(temp_path, "config_test.yaml"), "w") as f:
        yaml.dump(slice_config_test, f)