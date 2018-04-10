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

import FlexDenseNet
# reload(FlexDenseNet)
from FlexDenseNet import DenseNetMulti

# plt.ion()   # interactive mode

use_gpu = torch.cuda.is_available()
torch.cuda.is_available()

def visualize_model(model, num_images=6):
    images_so_far = 0
    fig = plt.figure()

    for i, data in enumerate(dataloaders['val']):
        inputs, labels = data
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs); print(outputs)
        _, preds = torch.max(outputs.data, 1)
#         print(_, preds)
        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('predicted: {}'.format(class_names[preds[j]]))
            imshow(inputs.cpu().data[j])

            if images_so_far == num_images:
                return

