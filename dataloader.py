import torchvision.transforms as transforms
import torch
import torch.utils.data as data
from torch.utils.data import random_split
import numpy as np
from PIL import Image
import glob



def loads(data_dir,ext,label):
    convert_tensor = transforms.ToTensor()

    ## Resize every image as specified in the paper
    resizer = transforms.Resize((256,256))

    #Normalized using the regular ImageNet values
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    
    # At least for X-ray greyscale means 1 channel, Encoder block requires 3 channels thus transform
    apply_greyscale = transforms.Grayscale(num_output_channels=3)

    image_list = []
    for filename in glob.glob(data_dir+'*.'+ext)[0:128]: #For development limit to two batches
        im=Image.open(filename)
        im = resizer(im)
        im = apply_greyscale(im) #todo are all inputs greyscale/1-channel?
        im = convert_tensor(im)
        im = normalize(im) 
        image_list.append((im,label))
    return image_list


def load(data_dir,batch_size=128, num_workers=4):

    train_dir = data_dir+"train/NORMAL/"
    val_dir_0 = data_dir+"val/NORMAL/"
    val_dir_1 = data_dir+"val/PNEUMONIA/"
    test_dir_0 = data_dir+"test/NORMAL/"
    test_dir_1 = data_dir+"test/PNEUMONIA/"

    train_dataset = loads(train_dir,"jpeg",0)
    val_dataset = loads(val_dir_0,"jpeg",0)+loads(val_dir_1,"jpeg",1)
    test_set = loads(test_dir_0,"jpeg",0)+loads(test_dir_1,"jpeg",1)



    train_loader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        drop_last=False)
    test_loader = data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        drop_last=False)

    return train_loader, val_loader, test_loader

