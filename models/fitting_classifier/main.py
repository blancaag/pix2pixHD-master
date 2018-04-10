##########
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --name testv0_vggbn_bs128_f --checkpoint '' --batch_size 128 
# --model vgg19 --lr 0.00002 --nepochs 30 --lr_decay 10 --finetune

"""
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python main.py --name testv0_vggbn_bs128_f --checkpoint testv1_bs512_do.2 --batch_size 128 --model densenet --lr 0.000001 --nepochs 30 --lr_decay 10
"""

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
# import importlib.util
# from importlib import reload
from imp import reload
import argparse

import networks
from options import Options
from train import *
from test import *

# # for loading Python 2 generated model-files
# from functools import partial
# import pickle
# pickle.load = partial(pickle.load, encoding="latin1")
# pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")

use_gpu = torch.cuda.is_available()
print('Num. of cuda visible devices: %d' %torch.cuda.device_count(), list(range(torch.cuda.device_count())))   

# Loading options
opt = Options().parse()

# Setting the model

if opt.model == 'densenet':
        model = networks.DenseNetMulti(nchannels=opt.nc_input, drop_rate=opt.drop_out)
        model.initialize(opt)
        
elif opt.model == 'vgg19':
        model = networks.vgg19_bn(pretrained=True)
        model.initialize(opt)
        opt.nc_input = 3
        
if torch.cuda.is_available():
        model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count()))).cuda()

print(model)

# Setting the data loaders
from data_loading import *
datasets, dataloaders = set_dataloaders(opt)
datasets_sizes = {x: len(datasets[x]) for x in opt.phases}

# Setting the loss and training

criterion1 = nn.CrossEntropyLoss()
criterion2 = nn.BCELoss()
criterion3 = nn.BCEWithLogitsLoss()
criterion4 = nn.MSELoss()

if opt.phase == 'train':
        model = train_model(model, dataloaders, criterion3, opt, datasets_sizes)                  
else: 
        print('Add test code from notebook')

# ### TESTING ###
# phases = ['test']
# data_dir = '/blanca/datasets/2nd_FLLC_MB/output/trainset/preprocessed_ilstack'
# data_dir = '.../datasets/2nd_FLLC_MB/output/trainset/preprocessed_ilstack'
#
# if opt.test_data: data_dir = opt.test_data
#
# datasets, dataloaders = set_dataloaders(data_dir, batch_size, nworkers, phases)
#
# test_model(model, dataloaders, phases) #### add output folders inside output