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
        for label, path in self.file_list:
            img = Image.open(path)
            img = preprocess_img(img)
            image_list.append((img, label))
        self.image_list = image_list

    def _get_file_list(self):
        file_list = []
        label_list = []

        # set split back to 'train'
        if self.split == 'train_complete':
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
            path0 = (0, split+'/NORMAL/')
            path1 = (1, split+'PNEUMONIA/')
            paths = [path0, path1]

        if self.split == 'train':
            paths = [path0]

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

#def loads(data_dir,ext,label):
#    convert_tensor = transforms.ToTensor()
#
#    ## Resize every image as specified in the paper
#    resizer = transforms.Resize((256,256))
#
#    #Normalized using the regular ImageNet values
#    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                 std=[0.229, 0.224, 0.225])
#    
#    # Encoder block requires 3 channels thus transform
#    apply_greyscale = transforms.Grayscale(num_output_channels=3)
#    #print(len(glob.glob(data_dir+'*.'+ext)))
#
#    image_list = []
#    for filename in tqdm(glob.glob(data_dir+'*.'+ext)):
#        im=Image.open(filename)
#        im = resizer(im)
#        im = apply_greyscale(im) 
#        im = convert_tensor(im)
#        im = normalize(im)
#        image_list.append((im,label))
#        del im
#    return image_list


def load(data_dir,batch_size=64, num_workers=4):

    # if statements can be removed
    if data_dir == "chest_xray":
        # preload=True not tested yet
        train_dataset = LoadDataset(data_dir, 'train', preload=True)

        val_dataset = LoadDataset(data_dir, 'val', preload=True)
        test_dataset = LoadDataset(data_dir, 'test', preload=True)

        train_dataset_complete = LoadDataset(data_dir, 'train_complete', preload=True)
    elif data_dir == "OCT2017":
        train_dataset = LoadDataset(data_dir, split='train')

        val_dataset = LoadDataset(data_dir, split='val')

        test_dataset = LoadDataset(data_dir, split='test')

        train_dataset_complete = LoadDataset(data_dir, split='train_complete')

    ## As we use Nvidia GPU's pin_memory for speedup using pinned memmory
    train_loader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    
    val_loader = data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False, pin_memory=False)
    
    test_loader = data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False, pin_memory=False)

    train_complete = data.DataLoader(
        train_dataset_complete, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False, pin_memory=False)
    
    return train_loader, train_complete, val_loader, test_loader #, [test_loader_normal, test_loader_abnormal]

if __name__ == "__main__":
    load('OCT2017')