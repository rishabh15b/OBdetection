import cv2
import os
import numpy as np

import albumentations as albu

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    CLASSES = ['background','stockpile']
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)


def get_training_augmentation():
    train_transform = [
        # albu.Resize(1024, 1024),
        albu.RandomBrightness(p=0.3),
        albu.RandomContrast(p=0.3),
        # albu.HueSaturationValue(p=0.3),
        # albu.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        # albu.ColorJitter(p=0.3),
        # albu.CutMix(num_holes=8, p=0.5),  # Adjust parameters as needed
        # albu.RandomShadow (p=0.3),
        albu.RandomRotate90(p=2),
        albu.Affine(scale=0.5, p=0.2),
        

        # albu.HueSaturationValue(p=0.3),
        # albu.GridDistortion (num_steps=5, distort_limit=0.3, p=0.2),
        # albu.ColorJitter(p=0.3),
        # albu.RandomGridShuffle (grid=(3, 3),p=0.3),
        # # albu.CutMix(num_holes=8, p=0.5),  # Adjust parameters as needed
        # albu.RandomRotate90(p=0.2),
        # # albu.Affine(scale=0.5, p=0.2),   #HSV gitter noise cut cutmix scheduler 
        # albu.PiecewiseAffine(scale=(0.03, 0.1), nb_rows=5, nb_cols=5, p=0.3),

        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        # albu.Resize(1024, 1024),
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)