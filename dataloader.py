import torchvision.transforms as transforms
import torch
import torch.utils.data as data
from torch.utils.data import random_split
import numpy as np
from PIL import Image, ImageFile
import glob
from torch.utils.data import Dataset
from tqdm import tqdm

import os

from pathlib import Path

from anomalib.data import TaskType
from anomalib.data.btech import BTech
from anomalib.data.mvtec import MVTec
from anomalib.data.utils import InputNormalizationMethod

# required for certain images in OCT2017
ImageFile.LOAD_TRUNCATED_IMAGES = True

# TODO: train per subset
def load_btad(split, subset, batch_size=64, num_workers=8):
    dataset_root = Path.cwd().parent / "datasets" / "BTech"

    btech_datamodule = BTech(
        root=dataset_root,
        category=subset,
        image_size=256,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=num_workers,
        task=TaskType.CLASSIFICATION,
        normalization=InputNormalizationMethod.NONE,  # don't apply normalization, as we want to visualize the images
    )
    # check if the data is available and setup
    btech_datamodule.prepare_data()
    btech_datamodule.setup()
    if split == 'train':
        return btech_datamodule.train_data
    elif split == 'test':
        return btech_datamodule.test_data
    elif split == 'val':
        return btech_datamodule.val_data
    else:
        raise NotImplementedError


# TODO: train per subset
def load_mvtec(split, subset, batch_size=64, num_workers=8):
    dataset_root = Path.cwd().parent / "datasets" / "MVTec"
    # MVTec Classification Train Set
    mvtec_datamodule = MVTec(
        root=dataset_root,
        category=subset,
        image_size=256,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=num_workers,
        task="classification",
        normalization=InputNormalizationMethod.NONE,  # don't apply normalization, as we want to visualize the images
    )
    # check if the data is available and setup
    mvtec_datamodule.prepare_data()
    mvtec_datamodule.setup()
    if split == 'train':
        return mvtec_datamodule.train_data
    elif split == 'test':
        return mvtec_datamodule.test_data
    elif split == 'val':
        return mvtec_datamodule.val_data
    else:
        raise NotImplementedError

class LoadDataset(Dataset):
    def __init__(self, data_dir, split, ext='jpeg', subset=None, batch_size=64, num_workers=8):
        self.data_dir = data_dir
        self.ext = ext
        self.split = split
        self.num_workers = num_workers
    

        # relevant for BTAD and MVTEC datasets
        self.subset = subset
        self.batch_size = batch_size

        if (data_dir == 'btech') or (data_dir == 'mvtec'):
            self.preload_files()
            self.preload = True
        else:
            self.file_list, self.labels = self._get_file_list()            
            self.preload = False

    def preload_files(self):
        if self.data_dir == 'btech':
            self.image_list = load_btad(self.split, self.subset, batch_size=self.batch_size, num_workers=self.num_workers)
        elif self.data_dir == 'mvtec':
            self.image_list = load_mvtec(self.split, self.subset, batch_size=self.batch_size, num_workers=self.num_workers)
        else:
            image_list = []
            for label, path in zip(self.labels, self.file_list):
                img = Image.open(path)
                img = preprocess_img(img)
                image_list.append((img, label))
            self.image_list = image_list

    def _get_file_list(self):
        file_list = []
        label_list = []
        # set split to 'train'
        if self.split == 'train_abnormal':
            split = 'train'
        else:
            split = self.split

        if self.data_dir == 'OCT2017':
            path0 = (0, 'data/'+self.data_dir+'/'+split+'/NORMAL/')
            path1 = (1, 'data/'+self.data_dir+'/'+split+'/CNV/')
            path2 = (1, 'data/'+self.data_dir+'/'+split+'/DME/')
            path3 = (1, 'data/'+self.data_dir+'/'+split+'/DRUSEN/')
            paths = [path0, path1, path2, path3]
        elif self.data_dir == 'chest_xray':
            path0 = (0, 'data/'+self.data_dir+'/'+split+'/NORMAL/')
            path1 = (1, 'data/'+self.data_dir+'/'+split+'/PNEUMONIA/')
            paths = [path0, path1]

        if self.data_dir == 'mvtec':
            file_list, label_list = load_mvtec(self.split, subset=self.subset, batch_size=self.batch_size)
        elif self.data_dir == 'btech':
            file_list, label_list = load_btad(self.split, subset=self.subset, batch_size=self.batch_size)
        else:
            # OCT or chest_xray
            if self.split == 'train':
                paths = [path0]
            elif self.split == 'train_abnormal':
                if self.data_dir == 'OCT2017':
                    paths = [path1, path2, path3]
                else:
                    paths = [path1]
            for label, path in paths:
                for filename in glob.glob(path+'*.'+self.ext):
                    file_list.append(filename)
                    label_list.append(label)
        return file_list, label_list

    def __getitem__(self, index):
        if self.preload:
            img = self.image_list[index]['image']
            label = self.image_list[index]['label']

            # an extra check for when loading in data with anomalib
            # we need to make sure the train set only contains normal samples
            if (self.split == 'train') and (label != 0):
                raise Exception('Train contains non 0 labels. The train set should not contain any samples considered abnormal.')
        else:
            file_path = self.file_list[index]
            img = Image.open(file_path)
            img = preprocess_img(img)
            label = self.labels[index]
        return img, label
    
    def __len__(self):
        if self.preload:
            return len(self.image_list) 
        return len(self.file_list)

def preprocess_img(img):
    convert_tensor = transforms.ToTensor()

    ## Resize every image as specified in the paper
    resizer = transforms.Resize((256,256))

    #Normalized using the regular ImageNet values
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    apply_greyscale = transforms.Grayscale(num_output_channels=3)
    img = resizer(img)
    img = apply_greyscale(img) 
    img = convert_tensor(img)
    img = normalize(img)
    return img

def load(data_dir,batch_size=64, num_workers=4, subset=None):
    train_dataset = LoadDataset(data_dir, split='train', subset=subset, num_workers=num_workers)
    test_dataset = LoadDataset(data_dir, split='test', subset=subset, num_workers=num_workers)

    # only the test set is loaded into the dataloader
    test_loader = data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False, pin_memory=False)
   

    if data_dir in ['btech', 'mvtec']:
        train_abnormal = LoadDataset(data_dir, split='val', subset=subset)
        return train_dataset, train_abnormal, test_loader
    else:
        train_abnormal = LoadDataset(data_dir, split='train_abnormal', subset=subset)
        return train_dataset, train_abnormal, test_loader