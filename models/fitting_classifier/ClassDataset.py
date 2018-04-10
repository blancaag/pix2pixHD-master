import torch
import torchvision.transforms as transforms

import os, sys
from pathlib import Path
from functools import reduce
import operator
import cv2
import numpy as np
import random

def read_image_OpenCV(path, target_size=None):
        im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if target_size: im = cv2.resize(im, target_size)
        return im
        
def resize_image_OpenCV(im, target_size=None):
        im = cv2.resize(im, target_size)
        return im

def create_dataset_withLabel(path, classes=['bad_fit', 'good_fit'], return_pairs=None):
        path = Path(path)
        class_paths = [i for i in path.glob('*') if os.path.isdir(i)]
        pairs_path_list = []
        for i in class_paths:
                class_label = classes.index(i.parts[-1])
                print("Setting %s as %d: " %(i, class_label))
                pairs_path_list += create_dataset(i, class_label)
        
        print('shuffling...', end='')
        random.seed(1984)
        [random.shuffle(pairs_path_list) for i in range(int(1e2))]
        print('done')
        
        return pairs_path_list
        
def create_dataset(path, class_label=None, return_pairs=None):
        path = Path(path)
        ims_path_list = path.glob('*_m.png')
        
        pairs_path_list = []
        for i in ims_path_list:
                pair_name = i.parts[-1].split('_m.png')[0] + '.png'
                pair_path = str(list(path.glob(pair_name))[0])
                pair_path_mirror = str(i)
                pairs_path_list.append([(pair_path, class_label), (pair_path_mirror, class_label)])

        print('shuffling...', end='')
        random.seed(1984)
        [random.shuffle(pairs_path_list) for i in range(int(1e2))]
        print('done')
        
        if return_pairs==None: pairs_path_list = reduce(operator.add, pairs_path_list, [])
        return pairs_path_list

class ClassDataset(torch.utils.data.Dataset):
    
    def __init__(self):
        super(ClassDataset, self).__init__()
        
    def initialize(self, opt, phase):
        self.opt = opt
        self.root = opt.data_root 
        self.dataset_list = [phase]
        
        self.data_paths = []
        self.data_paths += create_dataset_withLabel(os.path.join(self.root, self.dataset_list[0])); 
            
    def apply_data_transforms(self, im, opt):
            
        transform_list = []
        transform_list += [
                    transforms.Lambda(lambda x: x[:, :, :self.opt.nc_input]),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5 for i in range(self.opt.nc_input)], [0.5 for i in range(self.opt.nc_input)])
                    ]                                 
                                         
        return transforms.Compose(transform_list)(im)
    
    def __getitem__(self, index):                             
                        
        # read target image
        im_path = self.data_paths[index][0]
        label = self.data_paths[index][1]
        
        im = read_image_OpenCV(im_path, target_size=(224, 224))
        
        # create output tensor
        im_tensor = self.apply_data_transforms(im, self.opt) 
        
        # print('im shape', im.shape, label, 'mask values', np.min(im[:,:,3]), np.max(im[:,:,3]))
        # print('min max tensor values: ', im_tensor.min(), im_tensor.max())
        # im_ = im_tensor.numpy().transpose(1, 2, 0)
        # print('im_ shape', im_.shape, label, 'mask values', np.min(im_[:,:,3]), np.max(im_[:,:,3]))
        
        # good/bad fit label
        label_tensor = torch.FloatTensor([label])

        return im_tensor, label_tensor
    
    def __get_x_item__(self, index):
        return self.__getitem__(index)
    
    def __len__(self):
        return len(self.data_paths)

    def name(self):
        return 'ClassDataset'