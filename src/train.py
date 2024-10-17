import torch
import os
from pathlib import Path
import yaml
import sys
import shutil
from yaml import SafeLoader
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from utils.plot import IOU_plot, Loss_plot, visualize
from utils.dataset import Dataset, get_training_augmentation, get_validation_augmentation, to_tensor, get_preprocessing
from utils.helper import model_definition, move_files
from sklearn.model_selection import train_test_split
import pandas as pd
from utils.helper import delete_files_in_directory
from logger import get_logger
from constants import Constant
from error import Errors
import warnings

warnings.filterwarnings("ignore")

LOGGER_ML = get_logger("Training Logs")

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

with open(ROOT / Constant.CONFIG.value) as f:
    slice_config = yaml.load(f, Loader=SafeLoader)

def train_model():

    resume_training = False  # Set to True if you want to resume training
    start_epoch = 0

    _,_,_,Arc = model_definition()

    # Collecting IOU scores for training and validation
    train_iou_scores = []
    valid_iou_scores = []
    train_losses = []
    valid_losses = []

    columns = ['Epoch', 'Train Loss', 'Valid Loss', 'Train IOU Score', 'Valid IOU Score']
    epoch_data = pd.DataFrame(columns=columns)
    max_score = 0
    patience = 10
    counter = 0

    checkpoint_path_final = os.path.join(model_path, f'{encoder}_{Arc}_final_model.pth')

    if resume_training:
        # Load the model, optimizer state, and start_epoch value from the checkpoint file
        checkpoint = torch.load(checkpoint_path_final)  # Provide the path to your checkpoint file
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']

    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

    for i in range(start_epoch, epochs):
        # print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        
        train_iou_scores.append(train_logs['iou_score'])  # Assuming 'iou_score' is available in train_logs
        valid_iou_scores.append(valid_logs['iou_score'])  # Assuming 'iou_score' is available in valid_logs

        train_losses.append(train_logs['dice_loss'])  # Assuming 'loss' is available in train_logs
        valid_losses.append(valid_logs['dice_loss'])  # Assuming 'loss' is available in valid_logs

        # Save the model if validation performance improves
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, f'{model_path}/best_model.pth')
            counter = 0  # Reset the counter
            # print('Model saved!')
        else:
            counter += 1
            LOGGER_ML.info(f'No improvement for {counter} epoch(s)')

            # Check for early stopping
            if counter >= patience:
                LOGGER_ML.info(f'Early stopping! No improvement for {patience} epochs.')
                break  # Stop training if no improvement for 'patience' epochs

        # Learning rate scheduling
        scheduler.step(valid_logs['iou_score'])
        
        # Save the model every 5 epochs
        if (i + 1) % 5 == 0 or i == epochs - 1:
            checkpoint_path = os.path.join(model_path, f'{encoder}_{Arc}_{i}.pth')

            torch.save(model , checkpoint_path)
            LOGGER_ML.info(f'Checkpoint saved at epoch {i}.')

            IOU_plot(train_iou_scores,valid_iou_scores, f"{plot_path}/iou_score_plot_{i}.png" )
            Loss_plot(train_losses,valid_losses, f"{plot_path}/loss_plot_{i}.png" )
        
        new_data = pd.DataFrame({
            'Epoch': [i],
            'Train Loss': [train_logs['dice_loss']],
            'Valid Loss': [valid_logs['dice_loss']],
            'Train IOU Score': [train_logs['iou_score']],
            'Valid IOU Score': [valid_logs['iou_score']]
            })

        epoch_data = pd.concat([epoch_data, new_data], ignore_index=True)

        epoch_data.to_csv(f"{model_path}/epoch_data.csv", index=False)

    torch.save(model, checkpoint_path_final)

    IOU_plot(train_iou_scores,valid_iou_scores, f"{model_path}/iou_score_plot.png" )
    Loss_plot(train_losses,valid_losses, f"{model_path}/loss_plot.png" )

    # return counter

if __name__ == "__main__":

    experiment_path = os.path.join(slice_config['train']['experiment_path'],slice_config['train']['experiment_name'])
    isExist = os.path.exists(experiment_path)
    if not isExist:
        os.makedirs(experiment_path)
   
    images_folder = slice_config['data_preparation']['processed_images_dir']
    masks_folder = slice_config['data_preparation']['processed_masks_dir']

    model_path = slice_config['train']['model_path']
    isExist = os.path.exists(model_path)
    if not isExist:
        os.makedirs(model_path)

    plot_path = slice_config['train']['model_plots']
    isExist = os.path.exists(plot_path)
    if not isExist:
        os.makedirs(plot_path)


    x_train_dir = slice_config['train']['train_dir']
    y_train_dir = slice_config['train']['train_annotations_dir']

    x_valid_dir = slice_config['train']['val_dir']
    y_valid_dir = slice_config['train']['val_annotations_dir']

    # x_test_dir = os.path.join(data_path, 'test')
    # y_test_dir = os.path.join(data_path, 'testannot')

    if not (os.path.exists(x_train_dir) and os.path.exists(y_train_dir) and
            os.path.exists(x_valid_dir) and os.path.exists(y_valid_dir)): #and
            # os.path.exists(x_test_dir) and os.path.exists(y_test_dir)):

        image_files = os.listdir(images_folder)
        mask_files = os.listdir(masks_folder)

        # x_temp, x_test, y_temp, y_test = train_test_split(image_files, mask_files, test_size=6, random_state=42)
        x_train, x_valid, y_train, y_valid = train_test_split(image_files, image_files, test_size=0.2, random_state=42)

        move_files(images_folder, x_train_dir, x_train)
        move_files(masks_folder, y_train_dir, y_train)

        move_files(images_folder, x_valid_dir, x_valid)
        move_files(masks_folder, y_valid_dir, y_valid)

        # move_files(os.path.join(data_path, images_folder), x_test_dir, x_test)
        # move_files(os.path.join(data_path, masks_folder), y_test_dir, y_test)
    else:
        LOGGER_ML.info("Data has already been split. Skipping the data splitting step.")
    training_data_len = len(os.listdir(x_train_dir))
    validation_data_len = len(os.listdir(x_valid_dir))
    delete_files_in_directory(images_folder)
    delete_files_in_directory(masks_folder)


    device = slice_config['train']['device']
    epochs = slice_config['train']['epochs']
    lr = slice_config['train']['lr']
    CLASSES = slice_config['train']['classes']
    batch_size = slice_config['train']['batch_size']

    _, encoder, encoder_weights, _ = model_definition()
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)
        

    train_dataset = Dataset(
        x_train_dir, 
        y_train_dir, 
        augmentation=get_training_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    valid_dataset = Dataset(
        x_valid_dir, 
        y_valid_dir, 
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    model, _, _, _ = model_definition()

    loss = smp.utils.losses.DiceLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5)
    ]

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=lr),
    ])

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=device,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        device=device,
        verbose=True,
    )

    LOGGER_ML.info(
    "Training Params:\n"
    "Encode_name: {}, Encoder_weights: {}, Epochs: {}, LR: {}, Batch_Size: {}, Train_Size: {}, Valid_Size: {}"
    .format(encoder, encoder_weights, epochs, lr, batch_size, training_data_len, validation_data_len)
    )

    train_model()

    shutil.move(model_path, experiment_path)
    with open(os.path.join(experiment_path, "config_train.yaml"), "w") as f:
        yaml.dump(slice_config, f)

    