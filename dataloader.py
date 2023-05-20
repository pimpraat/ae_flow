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

# required for certain images in OCT2017
ImageFile.LOAD_TRUNCATED_IMAGES = True

# TODO: train per subset
def load_btad(split, subset=None, n_abnormal_samples_per_class=10):
    file_list = []
    label_list = []
    
    datasets = ['01', '02', '03']
    exts = ['bmp', 'png', 'bmp']
    dataset_to_exts = {dataset:ext for dataset, ext in zip(datasets, exts)}

    # when we are only using a subset, filter out that subset and its corresponding ext
    if subset != None:
        datasets = [subset]
        exts = [dataset_to_exts[subset]]

    # if validation split: take n_abnormal_samples of each class for each dataset
    if split in ['test', 'train_abnormal']:

        # path to dirs
        for dataset, ext in zip(datasets, exts):
            path0 = (0, f'data/btad/{dataset}/test/ok/')
            path1 = (1, f'data/btad/{dataset}/test/ko/')
            paths = [path0, path1]

            # select 2 samples of each class for each dataset
            # currently does the same for both splits (as we do not use a validation set but cross validate)
            if split == 'train_abnormal':
                paths = [path1]
                for label, path in paths:
                    # one path is one class, reset files_added
                    files_added = 0
                    for filename in glob.glob(path+'*.'+ext):
                        if files_added < n_abnormal_samples_per_class:
                            file_list.append(filename)
                            label_list.append(label)
                            files_added += 1
            # for the test set, select everything that has not been used in training or validation (above)
            else:
                for label, path in paths:
                    files_added = 0
                    for filename in glob.glob(path+'*.'+ext):
                        # only start adding to file list after the first two have
                        if files_added >= n_abnormal_samples_per_class:
                            file_list.append(filename)
                            label_list.append(label)
                        else:
                            files_added += 1

            
    # else train
    # btad has no abnormal samples for train
    else:
        for dataset, ext in zip(datasets, exts):
            path = f'data/btad/{dataset}/train/ok/'
            for filename in glob.glob(path+'*.'+ext):
                file_list.append(filename)
                label_list.append(0)

    return file_list, label_list

# TODO: train per subset
def load_mvtec(split, subset=None, n_abnormal_samples=20):
    file_list = []
    label_list = []
    #print(os.getcwd())
    root_dir = 'data/mvtec/'
    subsets = os.listdir(root_dir) 

    # if we are only interested in a subset
    if subset != None:
        subsets = [subset]

    # each subset is a category of items in mvtec
    for subset in subsets:

        # we want to select n_abnormal_samples from each category
        n_abnormal_samples_selected = 0
        if split != 'train':
            path = root_dir+subset+'/test/'

            subdirs = os.listdir(path)
            #abnormal_dirs.remove('good')
            for dir in subdirs:
                for filename in glob.glob(path+dir+'/*.png'):

                    # for abnormal set, we check all folders except for good
                    if (split == 'train_abnormal') and (n_abnormal_samples_selected < n_abnormal_samples) and dir != 'good':
                        file_list.append(filename)
                        label_list.append(1)
                        n_abnormal_samples_selected += 1
                    elif (split == 'test') and (n_abnormal_samples_selected >= n_abnormal_samples):
                        file_list.append(filename)
                        if dir != 'good':
                            label = 1
                        else:
                            label = 0
                        label_list.append(label)
                    # when split = test, the number of abnormal samples will pass, in order to split test from train_abnormal
                    else:
                        n_abnormal_samples_selected += 1
        else:
            path = root_dir+subset+'/train/good/'
            subdirs = os.listdir(path)
            
            for filename in glob.glob(path+'*.png'):
                file_list.append(filename)
                label_list.append(0)
    return file_list, label_list
    

class LoadDataset(Dataset):
    def __init__(self, data_dir, split, ext='jpeg', preload=False, subset=None):
        self.data_dir = data_dir
        self.ext = ext
        self.split = split
        self.preload = preload
        
        # relevant for BTAD and MVTEC datasets
        self.subset = subset
        self.file_list, self.labels = self._get_file_list()            
        if preload:
            self.preload_files()

    def preload_files(self):
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

        elif self.data_dir == 'mvtec':
            file_list, label_list = load_mvtec(self.split, subset=self.subset)
        elif self.data_dir == 'btad':
            file_list, label_list = load_btad(self.split, subset=self.subset)
        else:
            # non bean tech datasets
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

        print(self.split)
        print(len(file_list))
        print(len(label_list))
        return file_list, label_list

    def __getitem__(self, index):
        if self.preload:
            img = self.image_list[index][0]
            label = self.image_list[index][1]
        else:
            file_path = self.file_list[index]
            img = Image.open(file_path)
            img = preprocess_img(img)
            label = self.labels[index]
        return img, label
    
    def __len__(self):
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

def load(data_dir,batch_size=64, num_workers=4, return_dataloaders=True, subset=None):
    train_dataset = LoadDataset(data_dir, split='train', subset=subset)
    test_dataset = LoadDataset(data_dir, split='test', subset=subset)
    train_abnormal = LoadDataset(data_dir, split='train_abnormal', subset=subset)
    if not return_dataloaders:
        # only the test_dataset will be a dataloader
        test_loader = data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False, pin_memory=False)
        return train_dataset, train_abnormal, test_loader

    ## As we use Nvidia GPU's pin_memory for speedup using pinned memmory

    train_loader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False, pin_memory=False)
    train_abnormal_loader = data.DataLoader(
        train_abnormal, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False, pin_memory=False)
    return train_loader, train_abnormal_loader, test_loader
