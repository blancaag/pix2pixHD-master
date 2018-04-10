import torch.utils.data as data

from PIL import Image
import os
import os.path
import operator
from functools import reduce
from glob import glob
import cv2
import random

import traceback
import logging
import random

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

INPUT_EXTENSIONS = ['*.jpg', '*.JPG', '*.png']
INPUT_EXTENSIONS_RECURSIVE = ['**/*.jpg', '**/*.JPG', '**/*.png']

def is_image_file(filename):
    """Checks if a file is an image.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images

def make_list_of_paths_labels(root, mode, classes=None, split=None):

    if classes is None: 
        print(root, os.path.join(root, '**/*.png'))
        path_list = sorted(reduce(operator.add, [glob(os.path.join(root, i), recursive=True) 
                                                 for i in INPUT_EXTENSIONS_RECURSIVE], []))
        label = 'None'
        path_label_list = [(path, label) for path in path_list]
        print('HERE', len(path_list), len(path_label_list))

    else:
        # try:
        if True:
            path_label_list = []
            for i in classes:

                path_list = sorted(reduce(operator.add, 
                                   [glob(os.path.join(os.path.join(root, i), j), recursive=True) \
                                   for j in INPUT_EXTENSIONS_RECURSIVE], []))

                label = classes.index(i)

                '''fixed shuffle of the data'''
                random.seed(1984)
                for s in range(int(10e1)): random.shuffle(path_list)
    #             print(path_list[0], path_list[-1])

                '''splitting between train and val sets'''
                sidx = int(len(path_list) * split)
                if mode == 'train': data_split = path_list[:sidx]
                else: data_split = path_list[sidx:]

                [path_label_list.append((path, label)) for path in data_split]
       
    return path_label_list
       
    #     except Exception as e:
    #
    #         logging.error(traceback.format_exc())
    #
    #         path_label_list = []
    #         for i in classes:
    #             class_path = os.path.join(root, i)
    #             sub_folders = [j for j in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, j))]
    #
    #             path_list = sorted(reduce(operator.add,
    #                                [glob(os.path.join(os.path.join(class_path, j), k)) \
    #                                for k in INPUT_EXTENSIONS for j in sub_folders], []))
    #
    #             label = classes.index(i)
    #
    #             '''fixed shuffle of the data'''
    #             random.seed(1984)
    #             for s in range(int(10e3)): random.shuffle(path_list)
    #
    #             '''splitting between train and val sets'''
    #             sidx = int(len(path_list) * split)
    #             if 'train' in root: data_split = path_list[:sidx]
    #             else: data_split = path_list[sidx:]
    #
    #             [path_label_list.append((path, label)) for path in data_split]
    #
    # #     print(len(path_label_list))
    # #     [print(i) for i in path_label_list[0:10]]

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def opencv_loader(path):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return im
    
def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    print(get_image_backend)
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class DatasetFromSubFolders(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::
        root/class1/**/*.png
        root/class2/**/*.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, mode, data_transforms=None, label_transforms=None, loader=opencv_loader, mult_dd=False, split=0.8):
        
        if mode in ['train', 'val']: 
            self.classes = sorted([i for i in os.listdir(root) if os.path.isdir(os.path.join(root, i))])
            print(self.classes)
            for i in self.classes:
                classf_content = glob(os.path.join(root, '*'))
                print(classf_content)
                if any(os.path.isdir(j) for j in classf_content): mult_dd = True
            
            if mult_dd: dataset = make_list_of_paths_labels(root, mode, self.classes, split)
            else: print('TODO..')
            # else: dataset = make_list_of_paths_labels(root, self.classes, split)
        
        else: 
            self.classes = None
            dataset = make_list_of_paths_labels(root)
        
        if len(dataset) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(INPUT_EXTENSIONS)))

        self.root = root
        self.dataset = dataset
        self.data_transforms = data_transforms
        self.label_transforms = label_transforms
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, label = self.dataset[index]
        data = self.loader(path)

        if self.data_transforms is not None:
            try: data = self.data_transforms(data)
            except AttributeError as e:
                logging.error(traceback.format_exc())
                print(path, data)
                
        if self.label_transforms is not None:
            label = self.label_transforms(label)
        
        return path, data, label

    def __len__(self):
        return len(self.dataset)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str