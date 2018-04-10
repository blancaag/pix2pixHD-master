from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import models, transforms
# import matplotlib.pyplot as plt
import time
import os
from PIL import Image
# import importlib.util
# from importlib import reload
from imp import reload
import argparse

import DatasetFromSubFolders
# reload(DatasetFromSubFolders)
from DatasetFromSubFolders import *

use_gpu = torch.cuda.is_available()
torch.cuda.is_available()

#### FUNCTIONS FOR DATA PRE-PROCESSING

def resize_opencv(im, new_size):
    assert im.shape[:2] == (512, 512)
    im = cv2.resize(im, new_size)
    return im

def split_hstack(im, nchannels):
    assert im.shape[:2] == (512, 1024)
    im_o = im[:, :int(im.shape[1]/2), :(nchannels)]
    im_m = im[:, int(im.shape[1]/2):, :(nchannels)]
    return im_o

def to_grayscale(arr):
    "If arr is a color image (3D array), convert it to grayscale (2D array)."
    if len(arr.shape) >= 3: return average(arr[:,:,:3], -1)  # average over the last axis (color channels)
    else: return arr
    
# def clip_image():

def append_maps(lbda_x, distance=None, channel=None, gray=False):
    
    x, y = lbda_x[0], lbda_x[1]
    
    if distance is None: return x
    
    out1 = append_distance_map(x, x, y, ['euclidean'], channel=3)
    out2 = append_distance_map(out1, x, y, ['euclidean'], gray=True)
    
    return out
    
def append_distance_map(out, x, y, distance, channel=None, gray=False):
    """
    Appends to 'x' input an additional channel with the distance map between 'x' and 'y'.
    Returns a x[:,:,x.shape[2] + 1] dimenssional array.
    - x, y: numpy arrays of same dimensions
    - distance: lists of distances to calculate the distance maps
    - channels: specify the channel of the images the distance function is applied to
    """
    if gray: print(True); channel = None
    
    dmap = calculate_distance_map(x, y, channel, gray, losses=distance)
    dmap = dmap[distance[0]].reshape(dmap[distance[0]].shape + (1,))  
    
    if channel is None: print('HERE1'); xdmap = np.append(out, dmap, axis=2); print(xdmap.shape)
    else: print('HERE2'); xdmap = np.append(out[:,:,:channel], dmap, axis=2); print(xdmap.shape)
    
    return xdmap
    
def calculate_distance_map(x, y, channel, gray, losses=None, visualize=False):
    """
    Calculte a set of distance functions given two input images
    Returns a dictionary of {distance/loss: value}
    - losses (list): list of losses to use -should be any of the contained in below dictionary
    - nchannels (list): only if gray=False; indexing of the channels to use from the images
    - gray (bool): if transforming the images to grayscale before calculating the distances
    """
    _losses = {'euclidean': euclidean, 'manhattan': manhattan, 'KL': KL, 'minkowski': minkowski, 'cosine': cosine, 'ncc': ncc, 'hamming': hamming}
    
    if not losses: losses = _losses.keys()
    
    if gray and channel is None: 
        im, im_m = to_grayscale(x), to_grayscale(y)
        cmap='gray' # set color map to 'gray' or 'binary' for plt. visualization
    else: 
        im, im_m = x[:,:,channel], y[:,:,channel] 
        cmap=None
            
    im_mf = np.fliplr(im_m)
            
    # visualize the input images
    if visualize: plot_ims(np.array([im, im_mf]), cmap=cmap)
    
    for i in losses:
        # store the distances
        distances = {}
        if i not in _losses.keys(): print('Distance function not supported')
        distances[i] = _losses[i](im, im_mf, visualize, cmap)
    
#     [print('%s: %d  |' %(i, j)) for i, j in distances.items()]
#     print(distances)
    
    return distances

def set_dataloaders(data_dir, batch_size, nworkers, target_size, phases=['train', 'val'], nchannels=3):
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.Lambda(lambda x: split_hstack(x, nchannels)),
            transforms.Lambda(lambda x: resize_opencv(x, target_size)),
            # transforms.Lambda(lambda x: append_maps(x, channel=3)),
    #         transforms.Lambda(lambda (x, y): append_distance_map(x, y, ['euclidean'], 3, gray=True)),
    #         transforms.Scale(224),
    #         transforms.RandomSizedCrop(224),
    #         transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'val': transforms.Compose([
            transforms.Lambda(lambda x: split_hstack(x, nchannels)),
            transforms.Lambda(lambda x: resize_opencv(x, target_size)),
            # transforms.Lambda(lambda x: append_maps(x, channel=3)),
    #         transforms.Lambda(lambda (x, y): append_distance_map(x, y, ['euclidean'], 3, gray=True)),
    #         transforms.Scale(224),
    #         transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'test': transforms.Compose([
            transforms.Lambda(lambda x: split_hstack(x, nchannels)),
            transforms.Lambda(lambda x: resize_opencv(x, target_size)),
            transforms.ToTensor()
        ])
    }

    datasets = {x: DatasetFromSubFolders(data_dir, x, data_transforms[x]) for x in phases}

    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], 
                                                  batch_size=batch_size,
                                                  pin_memory=True,
                                                  shuffle=True, 
                                                  num_workers=nworkers) for x in phases}
                                       
    return datasets, dataloaders