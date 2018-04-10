###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################
import torch.utils.data as data
from PIL import Image
import os
from pathlib import Path
from functools import reduce
import operator
import random

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    print(dir, os.path.isdir(dir))
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images

def default_loader(path):
    return Image.open(path).convert('RGB')

class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)


####################################### CUSTOM DATA LOADING FUNTIONS
        
def reduce_and_shuffle_dict_values_nested1level(d):
    
    flat = reduce(operator.add,
                      [reduce(operator.add, i.values(), []) for i in list(d.values())], 
                      [])

    [random.shuffle(flat) for i in range(int(1e2))]

    return flat

def create_dataset_from_dir2subdir(dir, class_label=None, nitems=None):
    
    """
    Create a list of paths from a nesteed dir with two levels, selecting nitems from each dir of the last level 
    """
    
    EXT_RECURSIVE = ['**/*.jpg', '**/*.JPG', '**/*.png', '**/*.ppm']
    from collections import OrderedDict
        
    path = Path(dir)
    id_names = [i.parts[-1] for i in list(path.glob('*')) if os.path.isdir(i)]
    
    n_items_per_last_level = nitems

    data_dict = OrderedDict({i: {} for i in sorted(id_names)})
    data_dict_nitems = OrderedDict({i: {} for i in sorted(id_names)})

    # INITIALISE
    for i in id_names:
        for j in os.listdir(path/i):
            data_dict[i][j] = None

    # FILLING
    import random
    random.seed()

    for i in data_dict.keys():
        for j in data_dict[i].keys():
            txt_pl = reduce(
                  operator.add,
                  [list((path/i/j).glob('**/*.isomap.png'))],
                  [])

            # DICT WITH ALL PATHS
            data_dict[i][j] = txt_pl

            # DICT WITH MAX(N) PATHS
            random_idx = random.sample(range(len(txt_pl)), min(len(txt_pl), n_items_per_last_level))
            txt_pl_nitems = [str(txt_pl[i]) for i in random_idx]

            data_dict_nitems[i][j] = txt_pl_nitems

    print('Total found IDs in path %s: %d' %(path, len(data_dict_nitems)), '.. and selected %d per ID' %n_items_per_last_level)
    
    data_list_n_shuffled = reduce_and_shuffle_dict_values_nested1level(data_dict_nitems)
    data_list_n_shuffled_labeled = [(i, class_label) for i in data_list_n_shuffled]
    return data_list_n_shuffled_labeled

def create_dataset_fromIDsubfolders(path, nitems=None):
    assert os.path.isdir(path), '%s is not a valid directory' % path
    images = create_dataset_from_dir2subdir(path, nitems)
    return images

def create_dataset_fromIDsubfolders_withLabel(path, classes=['bad_fit', 'good_fit'], nitems=None):
    assert os.path.isdir(path), '%s is not a valid directory' % path
    
    path = Path(path)
    class_paths = [i for i in path.glob('*') if os.path.isdir(i)]
    # there are no pairs for videos (f.d.m.)
    path_list = []
    for i in class_paths:
            class_label = classes.index(i.parts[-1])
            path_list += create_dataset_from_dir2subdir(i, class_label, nitems)
    print('shuffling...', end='')
    random.seed(1984)
    [random.shuffle(path_list) for i in range(int(1e2))]
    print('done')
    
    return path_list

def create_dataset_withLabel(path, classes=['bad_fit', 'good_fit'], return_pairs=None):
        path = Path(path)
        class_paths = [i for i in path.glob('*') if os.path.isdir(i)]
        pairs_path_list = []
        for i in class_paths:
                class_label = classes.index(i.parts[-1])
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
