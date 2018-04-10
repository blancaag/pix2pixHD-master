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

use_gpu = torch.cuda.is_available()
torch.cuda.is_available()

def test_model(model, threshold, num_images=8, visualize=False):
    
    for param in self.parameters(): 
            print(param.requires_grad)
            param.requires_grad = False
    
    images_so_far = 0
    fig = plt.figure()   

    gf_pl = []
    bf_pl = []
    
    for i, data in enumerate(dataloaders['test']):

        paths, inputs, labels = data
    
        if use_gpu: inputs = Variable(inputs.cuda())
        else: inputs = Variable(inputs)

        out = model(inputs).data.cpu().numpy()
#         _, preds = torch.max(outputs.data, 1)
        preds = [1 if i[1] > threshold else 0 for i in out]
        
#         print(paths, out, preds)
        
        op = 'output'
        op_l = []
        for i in class_names: exec('%s_op = os.path.join(op, i); op_l.append(%s_op)' %(i, i))
                
        [os.makedirs(i) for i in op_l if not os.path.exists(i)]
        
        gf_pl.append([(i, list(j), k) for i, j, k in zip(paths, out, preds) if k == 1])
        bf_pl.append([(i, list(j), k) for i, j, k in zip(paths, out, preds) if k == 0])
                    
#         [os.system('cp %s %s' %(i, 'output/goodfit')) for i in gf_pl]
#         [os.system('cp %s %s' %(i, 'output/badfit')) for i in bf_pl]
        
        # visualize some results
        if visualize:
            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images: visualize == False
                
    return gf_pl, bf_pl

# gf_pl, bf_pl = test_model(model, threshold=0) # with 0.5: 94 1085; 0.25: 276 903
# gf_pl = reduce(operator.add, gf_pl, [])
# bf_pl = reduce(operator.add, bf_pl, [])
# print(len(gf_pl), len(bf_pl))
