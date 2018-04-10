import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
# from data.image_folder import make_dataset
from PIL import Image
from pathlib import Path

from functools import reduce
import operator
import cv2
import numpy as np

opt_use_PIL = False;
opt_use_OCV = not opt_use_PIL
print('woohoo, using OPENCV', opt_use_OCV)
opt_data_in_subfolderds = False

def read_image_PIL(path, opt):
        im = Image.open(path).convert('RGB')
        im = AB.resize((opt.loadSize * 2, opt.loadSize), Image.BICUBIC)
        # AB = transforms.ToTensor()(AB)
        return im

def read_image_OpenCV(path, opt, is_pair=True):
        im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if is_pair: im = cv2.resize(im, (opt.loadSize * 2, opt.loadSize))
        else: im = cv2.resize(im, (opt.loadSize, opt.loadSize))
        # AB = transforms.ToTensor()(AB)
        return im

def reduce_and_shuffle_dict_values_nested1level(d):

    flat = reduce(operator.add,
                      [reduce(operator.add, i.values(), []) for i in list(d.values())], 
                      [])

    [random.shuffle(flat) for i in range(int(1e4))]

    return flat

def create_dataset_from_dir2subdir(dir, nitems=None):
    
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
    
    return data_list_n_shuffled

def make_dataset_fromIDsubfolders(dir, nitems=None):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    images = create_dataset_from_dir2subdir(dir, nitems)
    return images

def fill_gaps(im, opt,
              fill_input_with=None,
              add_artificial=False,
              only_artificial=False,
              cm_p = '/blanca/resources/contour_mask/contour_mask.png'):
    
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
    
    if not only_artificial:
        for i in range(im.shape[2] - 1):
            if fill_input_with=='average':
                mask_average = im[:,:,3] != 0
                # print('Filling gaps with %f instead of %f' %(np.mean(im[:,:,0]), np.mean(im[:,:,0][mask_average])))
                im[:,:,i][~new_alpha] = (np.ones(im.shape[:2]) * np.mean(im[:,:,i][mask_average]))[~new_alpha]    
            else: im[:,:,i][~new_alpha] = which_fill[fill_input_with][~new_alpha]
    
    if add_artificial and opt.phase != 'test':
        ## SELECT AN OCCLUSSION AND APPLY TO THE IMAGE
        gpath = Path('/blanca')
        rpath = gpath / 'resources/db_occlusions'
        rpath = [rpath]
        rpaths = reduce(operator.add, 
                      [list(j.glob('*')) for j in rpath],
                      [])
        
#         reandom.seed()
        random_idx = random.sample(range(len(rpaths)), len(rpaths))
        ix = random.sample(range(len(rpaths)), 1)[0]
        # if opt.phase == 'test':
        #         rpath_test = '/blanca/resources/db_oclusions_test/000220_39.png'
        #         imr = read_image_OpenCV(rpath_test, opt, is_pair=False)
        # else:
        imr = read_image_OpenCV(str(rpaths[ix]), opt, is_pair=False)
        # imr = cv2.imread(str(rpaths[ix]), cv2.IMREAD_UNCHANGED)
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

def apply_data_transforms_pairs(im, which, opt):
        
    transform_list = []
    if which == 'B':
        transform_list += [
                transforms.Lambda(lambda x: fill_gaps(x, opt, fill_input_with='average')),
                transforms.Lambda(lambda x: x[:, :, :opt.output_nc]),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
    else:        
        transform_list += [
            transforms.Lambda(lambda x: fill_gaps(x, opt)),
            transforms.Lambda(lambda x: x[:, :, :opt.input_nc]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]                                    
                                         
    return transforms.Compose(transform_list)(im)

################# NON-NESTED INPUT FOLDER (ORIGINAL CODE)
import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images
################# 

class IDFolders_Pairs_Dataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        
        # PHASE
        self.dir_AB = opt.dataroot #os.path.join(opt.dataroot, opt.phase)
        
        ## FOLDER OR SUBFOLDERS
        n_items_perID = 10
        
        opt_data_in_subfolders = False
        if not opt_data_in_subfolders:
            self.AB_paths = sorted(make_dataset(self.dir_AB))
            assert(opt.resize_or_crop == 'resize_and_crop')
        else: 
            self.AB_paths = make_dataset_fromIDsubfolders(self.dir_AB, n_items_perID)
        
        print('Total dataset length:', len(self.AB_paths))
        
    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        
        use_PIL = self.opt.use_PIL
        use_OCV = not use_PIL
        
        if use_PIL: AB = read_image_PIL(AB_path, self.opt)
        if use_OCV: AB = read_image_OpenCV(AB_path, self.opt)
        
        # w_total = AB.size(2)
        # w = int(w_total / 2)
        # h = AB.size(1)
        # w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        # h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))
        #
        # A = AB[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
        # B = AB[:, h_offset:h_offset + self.opt.fineSize, w + w_offset:w + w_offset + self.opt.fineSize]
        #
        # A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
        # B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)
        
        input_nc = self.opt.label_nc if self.opt.label_nc != 0 else 3
        output_nc = self.opt.output_nc
        
        w = AB.shape[1] // 2
        A = AB[:, :w, :]
        B = AB[:, w:, :]

        A = apply_data_transforms_pairs(A, 'A', self.opt)
        B = apply_data_transforms_pairs(B, 'B', self.opt)

        # if (not self.opt.no_flip) and random.random() < 0.5:
        #     print('DO WE ENTER HERE?????????????') WE DO
        #     # SELECTING CHANNELS
        #     print(A.size(), A.size(2))
        #     idx = [i for i in range(A.size(2) - 1, -1, -1)]; print(idx)
        #     idx = torch.LongTensor(idx)
        #     A = A.index_select(2, idx)
        #     B = B.index_select(2, idx)

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)

        # A = A[:, :, :input_nc]
        # B = B[:, :, :output_nc]

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path, 'class_label': ''}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'


def apply_data_transforms(im, which, opt, nchannels):
        
    transform_list = []
    if which == 'B':
        transform_list += [
                transforms.Lambda(lambda x: fill_gaps(x, opt, fill_input_with='average')),
                transforms.Lambda(lambda x: x[:, :, :nchannels]),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
    else:        
        transform_list += [
            transforms.Lambda(lambda x: x.copy()),
            transforms.Lambda(lambda x: fill_gaps(x, opt, add_artificial=True)),
            transforms.Lambda(lambda x: x[:, :, :nchannels]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         
    ]                                    
                                         
    return transforms.Compose(transform_list)(im)


class IDFolders_Dataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        
        # PHASE
        self.dir_B = opt.dataroot #os.path.join(opt.dataroot, opt.phase)
        
        ## FOLDER OR SUBFOLDERS
        n_items_perID = 10
        
        opt_data_in_subfolders = False
        if not opt_data_in_subfolders:
            self.B_paths = sorted(make_dataset(self.dir_B))
            # assert(opt.resize_or_crop == 'resize_and_crop')
        else: 
            self.B_paths = make_dataset_fromIDsubfolders(self.dir_B, n_items_perID)
        
        print('Total dataset length:', len(self.B_paths))
        
    def __getitem__(self, index):
        B_path = self.B_paths[index]
        
        use_PIL = self.opt.use_PIL
        use_OCV = not use_PIL
        
        if use_PIL: B = read_image_PIL(B_path, self.opt, is_pair=False)
        if use_OCV: B = read_image_OpenCV(B_path, self.opt, is_pair=False)
        
        input_nc = self.opt.label_nc if self.opt.label_nc != 0 else 3
        output_nc = self.opt.output_nc
                
        A = apply_data_transforms(B, 'A', self.opt, input_nc)
        B = apply_data_transforms(B, 'B', self.opt, output_nc)

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)

        return {'A': A, 'B': B, 'A_paths': B_path, 'B_paths': B_path, 'class_label': ''}

    def __len__(self):
        return len(self.B_paths)

    def name(self):
        return 'AlignedDataset'