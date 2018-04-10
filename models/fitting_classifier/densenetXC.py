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

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default=None, help='checkpoint files dir')
parser.add_argument('--batch_size', default=4, help='if 12Gb-GPU: 32 elif 6Gb-GPU: 4')
parser.add_argument('--optimizer', default='adam', help='adam | SGD')
parser.add_argument('--nepochs', default=100, help='number of epocs')
parser.add_argument('--lr', default=1e-03, help='number of epocs')
parser.add_argument('--lr_decay', default=100, help='epochs step size for lr decay')
parser.add_argument('--beta1', default=0.9, help='Adam optimizer: beta1')
parser.add_argument('--beta2', default=0.999, help='Adam optimizer: beta2')
parser.add_argument('--phases', default=['train', 'val'], help='loop phases')
parser.add_argument('--train_data', help='trainning data dir')
parser.add_argument('--test_data', help='testing data dir')
opt = parser.parse_args()

args = vars(opt)
print('------------ Options -------------')
for k, v in sorted(args.items()): print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')


#### FUNCTIONS FOR DATA PRE-PROCESSING

class ResizeCV(object):
    """Resize the input PIL Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, new_shape, interpolation=Image.BILINEAR):
#         assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.new_shape = new_shape
        self.interpolation = interpolation

    def __call__(self, im): # h_w_shape
        """
        Resizes a numpy array
        Returns: im (np array) of dimensions im[new_shape[0],new_shape[1], :]
        - im (np array): array to be resized over the first two dimensions
        """
        im = cv2.resize(im, self.new_shape)
        return im

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

def split_hstack(im):
    assert im.shape[:2] == (512, 1024)
    im_o = im[:, :int(im.shape[1]/2), :]
    im_m = im[:, int(im.shape[1]/2):, :]
    return im_o, im_m

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

""" Distance functions """

def euclidean(x, y, v, m, mp=True):
    if v: print('SE'); plot_im(abs(x - y)**2, cmap=m)
    if mp: return abs(x - y)**2
    else: return np.sqrt(np.sum((x - y)**2))
              
def manhattan(x, y, v, m, mp=True):
    if v: print('AE'); plot_im(abs(x - y), cmap=m)
    if mp: return abs(x - y)
    else: return np.sum(abs(x - y))
              
def KL(x, y, v, m, mp=True):
    eps = 0.0000001
    if v: print('KL'); plot_im((x + eps) * np.log((x + eps) / (y + eps)), cmap=m)
    if mp: return (x + eps) * np.log((x + eps) / (y + eps))
    else: return np.sum((x + eps) * np.log((x + eps) / (y + eps)))

def minkowski(x, y, v, m, nroot=3, mp=True):
    def nth_root(x, n_root): return x ** (1/float(n_root))
    if v: print('ME%d' %nroot); plot_im(abs(x - y) ** nroot, cmap=m)
    if mp: return abs(x - y) ** nroot
    else: return nth_root(np.sum(abs(x - y) ** nroot), nroot)

def ncc(x, y, v, m, mp=True): 
    if v: print('NCC'); plot_im(((x - np.mean(x)) * (y - np.mean(y))) / ((x.size - 1) * np.std(x) * np.std(y)), cmap=m)
    if mp: return ((x - np.mean(x)) * (y - np.mean(y))) / ((x.size - 1) * np.std(x) * np.std(y))
    else: return np.sum((x - np.mean(x)) * (y - np.mean(y))) / ((x.size - 1) * np.std(x) * np.std(y))

def cosine(x, y, v=None, m=None):
    def square_rooted(x): return np.sqrt(np.sum([a*a for a in x]))
    num = np.sum(x * y)
    den = np.sqrt(np.sum(x ** 2)) * np.sqrt(np.sum(y ** 2))
    return num/den
 
def cosine_(x, y, v=None, m=None):
    import scipy
    return scipy.spatial.distance.cosine(x, y, w=None)
 
def jaccard(x, y, v=None, m=None):
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality / union_cardinality

def norm(x, y): return np.linalg.norm(x - y)

def hamming(x, y, v=None, m=None):
    assert len(x) == len(y)
    return np.sum(x != y)

def set_dataloaders(data_dir, batch_size, nworkers, phases=['train', 'val']):
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.Lambda(lambda x: split_hstack(x)),
            transforms.Lambda(lambda x: append_maps(x, channel=3)),
    #         transforms.Lambda(lambda (x, y): append_distance_map(x, y, ['euclidean'], 3, gray=True)),
            ResizeCV(target_size),

    #         transforms.Scale(224),
    #         transforms.RandomSizedCrop(224),
    #         transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Lambda(lambda x: split_hstack(x)),
            transforms.Lambda(lambda x: append_maps(x, channel=3)),
    #         transforms.Lambda(lambda (x, y): append_distance_map(x, y, ['euclidean'], 3, gray=True)),
            ResizeCV(target_size),

    #         transforms.Scale(224),
    #         transforms.CenterCrop(224),
            transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Lambda(lambda x: split_hstack(x)),
            transforms.Lambda(lambda x: x[1]),
            ResizeCV(target_size),
            transforms.ToTensor(),
        ])
    }


    datasets = {x: DatasetFromSubFolders(os.path.join(data_dir, x), data_transforms[x], mode=x) for x in phases}

    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], 
                                                  batch_size=batch_size,
                                                  pin_memory=True,
                                                  shuffle=True, 
                                                  num_workers=nworkers) for x in phases}
                                       
    return datasets, dataloaders



# data_dir = '/data_clean'
# target_size = (224, 224)
# batch_size = 16
# nworkers = 8
#
# datasets, dataloaders = set_dataloaders(data_dir, batch_size, nworkers)
# print(datasets['train'].classes)
# 
# def imshow(inp, title=None):
#     """Imshow for Tensor."""
#     print(inp.shape)
#     inp = inp.numpy().transpose((1, 2, 0))
#
#     cv2.imwrite('test.png', inp * 255)
#
#     plt.imshow(inp[:,:,[2,1,0]])
#     if title is not None:
#         plt.title(title)
#     plt.pause(0.001)  # pause a bit so that plots are updated
#
# # Get a batch of training data
# paths, inputs, labels = next(iter(dataloaders['train']))
#
# # Make a grid from batch
# out = torchvision.utils.make_grid(inputs[:,:3,:,:])
#
# class_names = datasets['train'].classes
#
# imshow(out, title=[class_names[x] for x in labels])

def train_model(model, dataloaders, criterion, optimizer, scheduler, nchannels, nepochs, phases):
    
    train_hist = {phase: {'loss': [], 'acc': []} for phase in phases}
    
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(nepochs):
        print('Epoch {}/{}'.format(epoch, nepochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in phases:
            if phase == 'train':
                scheduler.step(); print(scheduler.get_lr())
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                paths, inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                acc = torch.sum(preds == labels.data) / batch_size
                
#                 print('Batch loss/acc: %f/%f' %(loss, acc))
                
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)
                
                train_hist[phase]['loss'].append(loss.data[0])
                train_hist[phase]['acc'].append(acc)

            epoch_loss = running_loss / datasets_sizes[phase]
            epoch_acc = running_corrects / datasets_sizes[phase]
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_acc_loss = epoch_loss
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    torch.save(model.state_dict(), 
               os.path.join(checkpoint_dir, 'model_best_c%d_%f_%f_%d.pth.tar' %(nchannels, best_acc, best_acc_loss, int(time.time()/1000))))
    torch.save(model, 
               os.path.join(checkpoint_dir, 'model_best_c%d_%f_%f_%d.pt' %(nchannels, best_acc, best_acc_loss, int(time.time()/1000))))
    
#     model = torch.load('filename.pt')

#     if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
#         model.load_state_dict(checkpoint['state_dict'])
#     else:
#         model.load_state_dict(checkpoint)

    return model


def test_model(model, threshold, num_images=8, visualize=False):
    
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

""" MAIN """
# for loading Python 2 generated model-files
from functools import partial
import pickle
pickle.load = partial(pickle.load, encoding="latin1")
pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")

# Loading the model

data_dir = '/data_clean'
checkpoint_dir = 'checkpoints_densenet'
checkpoint_dir_ = 'checkpoints_densenet_'
target_size = (224, 224)
batch_size = int(opt.batch_size)
lr = float(opt.lr)
lr_decay = int(opt.lr_decay)
nworkers = batch_size
nepochs = int(opt.nepochs) #100
nichannels_densenet = 4
use_gpu = torch.cuda.is_available()

### TRAINING ##
phases = ['train', 'val']

# Setting the data loaders
datasets, dataloaders = set_dataloaders(data_dir, batch_size, nworkers, phases)
datasets_sizes = {x: len(datasets[x]) for x in phases}

# Setting the model
model = DenseNetMulti(nchannels=nichannels_densenet)
for param in model.parameters(): param.requires_grad = True

num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, 2)

criterion = nn.CrossEntropyLoss()
# criterion = nn.BCELoss()


if opt.optimizer == 'SGD': 
        optim.SGD(model.classifier.parameters(), lr=lr, momentum=0.9)
elif opt.optimizer == 'adam':
        optimizer = optim.Adam(model.classifier.parameters(), lr=lr, betas=(opt.beta1, 0.999))
else: print('Missing optimizer information')

if use_gpu: model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count()))).cuda()

""" Load model """
# checkpoint = os.path.join(checkpoint_dir_, 'model_best_c4_0.753670.pt')
# if opt.checkpoint: model = torch.load(opt.checkpoint, map_location={'cuda:0':'cuda:1'}) #map_location=lambda storage, loc: storage.cuda())
if opt.checkpoint: 
        # model = torch.load(opt.checkpoint)
        if torch.cuda.device_count() == 1: model.load_state_dict(torch.load(opt.checkpoint))
        else : model = torch.load(opt.checkpoint) #, map_location={'cuda:0':'cuda:1'}) #map_location=lambda storage, loc: storage.cuda())
        # model.load_state_dict(torch.load(opt.checkpoint))

# checkpoint = 'model_best_c4_75.2.pt'
# model = torch.load(checkpoint, map_location=lambda storage, loc: storage, pickle_module=pickle)
""" """
        
scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay, gamma=0.1)

print('Num. of cuda visible devices: %d' %torch.cuda.device_count(), list(range(torch.cuda.device_count())))

model = train_model(model, dataloaders, criterion, optimizer, scheduler, nichannels_densenet, nepochs, phases)

# for group in optimizer.param_groups:
#     print (group['lr'])
# #     group['lr'] = 0.0001
#     print (group['lr'])
#
#
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