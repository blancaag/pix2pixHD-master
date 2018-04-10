### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

# from data.base_dataset import BaseDataset, get_params, get_transform, normalize
# custom functions
from data.base_dataset import fill_gaps, read_image_OpenCV, resize_image_OpenCV, load_original_mask

# original function (used in testing mode)
from data.image_folder import make_dataset
# custom functions
from data.image_folder import create_dataset, create_dataset_withLabel, create_dataset_fromIDsubfolders, create_dataset_fromIDsubfolders_withLabel 

import os
import torch
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()
    def name(self):
        return 'BaseDataset'
    def initialize(self, opt):
        pass

class Pix2PixHDXDataset(BaseDataset):
        
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    
        self.dataset_list = opt.dataset_list

        self.target_paths = []
        if self.opt.isTrain:
            self.target_paths += create_dataset_withLabel(os.path.join(self.root, self.dataset_list[0])); print(len(self.target_paths))
            self.target_paths += create_dataset_withLabel(os.path.join(self.root, self.dataset_list[1])); print(len(self.target_paths))
            
            # self.target_paths += create_dataset_fromIDsubfolders_withLabel(os.path.join(self.root, self.dataset_list[1]), nitems=2); print(len(self.target_paths))
        else:
            # self.target_paths += create_dataset(os.path.join(self.root, self.dataset_list[0]))
            self.target_paths += make_dataset(os.path.join(self.root, self.dataset_list[0])); print(self.target_paths)
            
        self.dataset_size = len(self.target_paths) 
        
        if True: 
            self.images_mask_path = Path('../../../training_datasets/pix2pix/images_mask')
            self.images_mask_fname_list = [i.parts[-1] for i in list(self.images_mask_path.glob('*.png'))]
            
    def apply_data_transforms(self, im, which, opt, nchannels=None):
        
        transform_list = []
        if which == 'target':
            transform_list += [
                    transforms.Lambda(lambda x: fill_gaps(x, opt, fill_input_with='average')),
                    transforms.Lambda(lambda x: x[:, :, :nchannels]),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]
                
        elif which == 'targetXC':
             transform_list += [
                    transforms.Lambda(lambda x: x[:, :, :nchannels]),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5 for i in range(nchannels)], [0.5 for i in range(nchannels)])
                    ]
    
        elif which == 'input':        
            transform_list += [
                transforms.Lambda(lambda x: x.copy()),
                transforms.Lambda(lambda x: fill_gaps(x, opt, add_artificial=opt.isTrain)),
                # drops the compose&blended image alpha channel and loads the original one:
                transforms.Lambda(lambda x: load_original_mask(x, nchannels == 4, opt,
                                                               self.target_path, 
                                                               self.images_mask_fname_list, 
                                                               self.images_mask_path)),
                transforms.Lambda(lambda x: x[:, :, :nchannels]),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ]
            
        elif which == 'mask':        
            transform_list += [
                transforms.Lambda(lambda x: x.copy()),
                transforms.Lambda(lambda x: load_original_mask(x, nchannels == 4, opt,
                                                               self.target_path,
                                                               self.images_mask_fname_list,
                                                               self.images_mask_path)),
                transforms.Lambda(lambda x: fill_gaps(x, opt, only_extract_mask=True)),
                # transforms.Lambda(lambda x: resize_image_OpenCV(x, (224, 224))),

                transforms.Lambda(lambda x: x[:, :, 3]),
                transforms.Lambda(lambda x: 2*(x - np.max(x))/-np.ptp(x)-1),
                transforms.Lambda(lambda x: x / np.max(np.abs(x), axis=0)),
                    ]                                    
                                         
        return transforms.Compose(transform_list)(im)
    
    
    def __getitem__(self, index):                             
        
        target_tensor = target_label_tensor = inst_tensor = feat_tensor = target4C_tensor = target_mask_tensor = 0
        
        input_nc = self.opt.input_nc
        output_nc = self.opt.output_nc
        
        # read target image
        if self.opt.isTrain:
                self.target_path = self.target_paths[index][0]
                target_label = self.target_paths[index][1]
        else: self.target_path = self.target_paths[index]
        
        target_im = read_image_OpenCV(self.target_path, self.opt)
        target_im_resized = read_image_OpenCV(self.target_path, self.opt, target_size=(224, 224))
        
        # create input tensor first
        # if self.opt.isTrain:
        input_tensor = self.apply_data_transforms(target_im, 'input', self.opt, input_nc)
        # just for visualization purposes (check that the original alpha channel is loaded properly):
        input_mask_tensor = self.apply_data_transforms(target_im, 'mask', self.opt, input_nc)  

        # create output tensor
        if self.opt.isTrain: 
                # target image (image label) for the GAN
                target_tensor = self.apply_data_transforms(target_im, 'target', self.opt, output_nc)
                # target image with alpha channel
                target4C_tensor = self.apply_data_transforms(target_im_resized, 'targetXC', self.opt, nchannels=4)
                # good/bad fit label
                target_label_tensor = torch.FloatTensor([target_label])
        
        input_dict = {'input_image': input_tensor, 'input_mask': input_mask_tensor,
                      'target': target_tensor, 'label': target_label_tensor
              }

        return input_dict
    
    def __get_x_item__(self, index):
        return self.__getitem__(index)
    
    def __len__(self):
        return len(self.target_paths)

    def name(self):
        return 'Pix2PixHDXDataset'