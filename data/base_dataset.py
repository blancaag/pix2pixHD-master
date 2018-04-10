### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random

def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.resize_or_crop == 'resize_and_crop':
        new_h = new_w = opt.loadSize
    elif opt.resize_or_crop == 'scale_width_and_crop':
        new_w = opt.loadSize
        new_h = opt.loadSize * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.fineSize))
    y = random.randint(0, np.maximum(0, new_h - opt.fineSize))

    flip = random.random() > 0.5
    return {'crop_pos': (x, y), 'flip': flip}

def get_transform(opt, params, method=Image.BICUBIC, normalize=True):
    transform_list = []
    if 'resize' in opt.resize_or_crop:
        print('resizing2')
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Scale(osize, method))
    elif 'scale_width' in opt.resize_or_crop: # here
        # print('resizing')
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.loadSize, method)))

    if 'crop' in opt.resize_or_crop:
        print('cropping')
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.fineSize)))

    if opt.resize_or_crop == 'none':
        print('makepower')
        base = float(2 ** opt.n_downsample_global)
        if opt.netG == 'local':
            base *= (2 ** opt.n_local_enhancers)
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))

    if opt.isTrain and not opt.no_flip:
        # print('flipping')
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    
    return transforms.Compose(transform_list)

def normalize():
    print('Are we using this?')
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)

def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    # print(ow, oh, target_width)
    if (ow == target_width):
        return img
    w = target_width
    h = target_width # h = int(target_width * oh / ow)
    # print(w, h)
    return img.resize((w, h), method)

def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img

def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

###################### CUSTOM

from pathlib import Path
from functools import reduce
import operator
import cv2
import numpy as np

def read_image_PIL(path, opt):
        im = Image.open(path).convert('RGB')
        im = im.resize((opt.loadSize * 2, opt.loadSize), Image.BICUBIC)
        return im

def read_image_OpenCV(path, opt, is_pair=False, target_size=None):
        im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        # if is_pair: im = cv2.resize(im, (opt.loadSize * 2, opt.loadSize))
        ##
        # print(path)
        # print(im.shape)
        ##
        if target_size: im = cv2.resize(im, target_size)
        else: im = cv2.resize(im, (opt.loadSize, opt.loadSize))
        return im
        
def resize_image_OpenCV(im, target_size=None):
        # print(im.shape)
        im = cv2.resize(im, target_size)
        return im
        
# def reduce_and_shuffle_dict_values_nested1level(d):
#
#     flat = reduce(operator.add,
#                       [reduce(operator.add, i.values(), []) for i in list(d.values())],
#                       [])
#
#     [random.shuffle(flat) for i in range(int(1e4))]
#
#     return flat

# def create_dataset_from_dir2subdir(dir, nitems=None):
#
#     """
#     Create a list of paths from a nesteed dir with two levels, selecting nitems from each dir of the last level
#     """
#
#     EXT_RECURSIVE = ['**/*.jpg', '**/*.JPG', '**/*.png', '**/*.ppm']
#     from collections import OrderedDict
#
#     path = Path(dir)
#     id_names = [i.parts[-1] for i in list(path.glob('*')) if os.path.isdir(i)]
#
#     n_items_per_last_level = nitems
#
#     data_dict = OrderedDict({i: {} for i in sorted(id_names)})
#     data_dict_nitems = OrderedDict({i: {} for i in sorted(id_names)})
#
#     # INITIALISE
#     for i in id_names:
#         for j in os.listdir(path/i):
#             data_dict[i][j] = None
#
#     # FILLING
#     import random
#     random.seed()
#
#     for i in data_dict.keys():
#         for j in data_dict[i].keys():
#             txt_pl = reduce(
#                   operator.add,
#                   [list((path/i/j).glob('**/*.isomap.png'))],
#                   [])
#
#             # DICT WITH ALL PATHS
#             data_dict[i][j] = txt_pl
#
#             # DICT WITH MAX(N) PATHS
#             random_idx = random.sample(range(len(txt_pl)), min(len(txt_pl), n_items_per_last_level))
#             txt_pl_nitems = [str(txt_pl[i]) for i in random_idx]
#
#             data_dict_nitems[i][j] = txt_pl_nitems
#
#     print('Total found IDs in path %s: %d' %(path, len(data_dict_nitems)), '.. and selected %d per ID' %n_items_per_last_level)
#
#     data_list_n_shuffled = reduce_and_shuffle_dict_values_nested1level(data_dict_nitems)
#
#     return data_list_n_shuffled
#
# def make_dataset_fromIDsubfolders(dir, nitems=None):
#     assert os.path.isdir(dir), '%s is not a valid directory' % dir
#     images = create_dataset_from_dir2subdir(dir, nitems)
#     return images


def load_original_mask(im, activate, opt, im_path, images_mask_fname_list, images_mask_path):
    
    if activate:
            im_fname = im_path.split('/')[-1]
            if im_fname in images_mask_fname_list:
                mask_path = Path(images_mask_path, im_fname)
                mask = read_image_OpenCV(str(mask_path), opt)
                im[:,:,3] = mask
    return im

# if True:
#     images_mask_path = Path('training_datasets/pix2pix/images_mask')
#     images_mask_fname_list = [i.parts[-1] for i in list(images_mask_path.glob('*.png'))]
#
#     im_path = 'training_datasets/pix2pix/images_target_clean_classified/bad_fit/300w01_indoor_004_m.png'
#     im = read_image_OpenCV(im_path)
#     plot_im(im, alpha=True)
#     im = load_original_mask(im, im_path, images_mask_fname_list, images_mask_path)
#     plot_im(im, alpha=True)


def fill_gaps(im, opt,
              fill_input_with=None,
              add_artificial=False,
              only_artificial=False,
              only_extract_mask=False,
              load_original_alpha=False,
              cm_p = '../resources/contour_mask/contour_mask.png'):
    # print(cm_p)
    cm_p = '/media/dataserver/workspace/blanca/project/wip/pix2pixHDX-class-master/resources/contour_mask/contour_mask.png'
    # print(cm_p)
    if not fill_input_with: fill_input_with = opt.fill
    
    # creating fills for filling gaps
    n_fill_WB = cv2.randu(np.zeros(im.shape[:2]), 0, 255)
    n_fill_B = np.zeros(im.shape[:2]) * 255
    n_fill_W = np.ones(im.shape[:2]) * 255
    n_fill_G = np.ones(im.shape[:2]) * 127
    
    which_fill = dict(zip(['W', 'B', 'W&B', 'G'], [n_fill_W, n_fill_B, n_fill_WB, n_fill_G]))

    # read contour template mask
    # im_cm = cv2.imread(cm_p, cv2.IMREAD_UNCHANGED)
    im_cm = read_image_OpenCV(cm_p, opt, is_pair=False)
    assert im_cm is not None, 'Make sure there is a "contour_mask.png" file in this folder'
    mask = im_cm == 255 # contour mask: internal 
    
    new_alpha = im[:,:,3] != 0
    new_alpha[im_cm == 0] = 1 # ensuring we exclude corners what != txt. map
    im[:,:,3] = new_alpha * 255 # applying
    
    if only_extract_mask: return im
    
    # if not only_artificial is set, the (still) missing parts of the compose&blended image 
    # are filled with the average values of the image per channel
    if not only_artificial:
        for i in range(im.shape[2] - 1):
            if fill_input_with=='average':
                mask_average = im[:,:,3] != 0
                # print('Filling gaps with %f instead of %f' %(np.mean(im[:,:,0]), np.mean(im[:,:,0][mask_average])))
                im[:,:,i][~new_alpha] = (np.ones(im.shape[:2]) * np.mean(im[:,:,i][mask_average]))[~new_alpha]    
            else: im[:,:,i][~new_alpha] = which_fill[fill_input_with][~new_alpha]
    
    if add_artificial:
        ## SELECT AN OCCLUSSION AND APPLY IT TO THE IMAGE
        gpath = Path('..')
        rpath = gpath / 'resources/db_occlusions'
        rpath = [rpath]
        # print(rpath)
        
        gpath = Path('/media/dataserver/workspace/blanca/project/wip/pix2pixHDX-class-master')
        rpath =  gpath / 'resources/db_occlusions'
        rpath = [rpath]
        # print(rpath)
        
        rpaths = reduce(operator.add, 
                      [list(j.glob('*')) for j in rpath],
                      [])
        
        random_idx = random.sample(range(len(rpaths)), len(rpaths))
        ix = random.sample(range(len(rpaths)), 1)[0]
        imr = read_image_OpenCV(str(rpaths[ix]), opt, is_pair=False)

        alpha_artifitial = imr
        alpha_artifitial[alpha_artifitial != 0] = 1
        im[:,:,3][im[:,:,3] != 0] = 1
        new_alpha_artifitial = alpha_artifitial * im[:,:,3]
        
        # setting the new alpha channel
        im[:,:,3] = new_alpha_artifitial * 255
        
        # filling
        for i in range(im.shape[2] - 1):
            im[:,:,i][new_alpha_artifitial == 0] = which_fill[fill_input_with][new_alpha_artifitial == 0]
    
    return im
