import torchvision.transforms as transforms
import torch
import torch.utils.data as data
from torch.utils.data import random_split
import numpy as np
from PIL import Image, ImageFile
import glob
from torch.utils.data import Dataset
from tqdm import tqdm

# required for certain images in OCT2017
ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_btad(split):
    file_list = []
    label_list = []
    datasets = ['01', '02', '03']
    exts = ['bmp', 'png', 'bmp']

    n_abnormal_samples_per_class = 10

    # if validation split: take 2 of each class for each dataset
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

class LoadDataset(Dataset):
    def __init__(self, data_dir, split, ext='jpeg', preload=False):
        self.data_dir = data_dir
        self.ext = ext
        self.split = split
        self.preload = preload
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

        if self.data_dir == 'btad':
            file_list, label_list = load_btad(self.split)
        # non bean tech datasets
        else:
            if self.split == 'train':
                paths = [path0]
            elif self.split == 'train_abnormal':
                paths = [path1]
            for label, path in paths:
                for filename in glob.glob(path+'*.'+self.ext):
                    file_list.append(filename)
                    label_list.append(label)
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

def load(data_dir,batch_size=64, num_workers=4, return_dataloaders=True):
    train_dataset = LoadDataset(data_dir, split='train')
    test_dataset = LoadDataset(data_dir, split='test')
    train_abnormal = LoadDataset(data_dir, split='train_abnormal')
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
