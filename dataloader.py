################################################################################
# MIT License
#
# Copyright (c) 2022
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Autumn 2022
# Date Created: 2022-11-25
################################################################################

import torchvision
import torchvision.transforms as T
import torch
import torch.utils.data as data
from torch.utils.data import random_split
import numpy as np
from PIL import Image
import glob



def loads(data_dir,ext,label):
    convert_tensor = T.ToTensor()
    resizer = T.Resize((256,256))
    image_list = []
    for filename in glob.glob(data_dir+'*.'+ext): #assuming gif
        im=Image.open(filename)
        im = resizer(im)
        im = convert_tensor(im)
        image_list.append((im,label))
    return image_list


def load(data_dir,batch_size=128, num_workers=4):
    """
    Returns data loaders for 4-bit MNIST dataset, i.e. values between 0 and 15.

    Inputs:
        root - Directory in which the MNIST dataset should be downloaded. It is better to
               use the same directory as the part2 of the assignment to prevent duplicate
               downloads.
        batch_size - Batch size to use for the data loaders
        num_workers - Number of workers to use in the data loaders.
        download - If True, MNIST is downloaded if it cannot be found in the specified
                   root directory.
    """

    # Each data loader returns tuples of (img, label)
    # For the generative models we don't need the labels, which we need to take into account
    # when writing the train code.
    train_dir = data_dir+"train/NORMAL/"
    val_dir_0 = data_dir+"val/NORMAL/"
    val_dir_1 = data_dir+"val/PNEUMONIA/"
    test_dir_0 = data_dir+"test/NORMAL/"
    test_dir_1 = data_dir+"test/PNEUMONIA/"

    train_dataset = loads(train_dir,"JPEG",0)
    val_dataset = loads(val_dir_0,"JPEG",0)+loads(val_dir_1,"JPEG",1)
    test_set = loads(test_dir_0,"JPEG",0)+loads(test_dir_1,"JPEG",1)



    train_loader = data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        drop_last=False)
    test_loader = data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        drop_last=False)

    return train_loader, val_loader, test_loader

