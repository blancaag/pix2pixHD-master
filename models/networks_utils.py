### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
from .networks import *
import os

###############################################################################
# Functions
###############################################################################

def var(x, dim=1):
    '''
    Calculates variance.
    '''
    x_zero_meaned = x - x.mean(dim).expand_as(x)
    return x_zero_meaned.pow(2).mean(dim)
    
def normalize(x, dim=0):
    '''
    Projects points to a sphere.
    '''
    return x.div(x.norm(2, dim=dim).expand_as(x).clamp(min=1e-8))

def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    r"""Returns cosine similarity between x1 and x2, computed along dim.
    .. math ::
        \text{similarity} = \dfrac{x_1 \cdot x_2}{\max(\Vert x_1 \Vert _2 \cdot \Vert x_2 \Vert _2, \epsilon)}
    Args:
        x1 (Variable): First input.
        x2 (Variable): Second input (of size matching x1).
        dim (int, optional): Dimension of vectors. Default: 1
        eps (float, optional): Small value to avoid division by zero.
            Default: 1e-8
    Shape:
        - Input: :math:`(\ast_1, D, \ast_2)` where D is at position `dim`.
        - Output: :math:`(\ast_1, \ast_2)` where 1 is at position `dim`.
    Example::
        >>> input1 = autograd.Variable(torch.randn(100, 128))
        >>> input2 = autograd.Variable(torch.randn(100, 128))
        >>> output = F.cosine_similarity(input1, input2)
        >>> print(output)
    """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    
    return w12 / (w1 * w2).clamp(min=eps)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_G(input_nc, output_nc, ngf, netG, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1,
             n_blocks_local=3, norm='instance', gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    
    # default: global
    if netG == 'global': netG = GlobalGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)
    elif netG == 'local': netG = LocalEnhancer(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global,
                                  n_local_enhancers, n_blocks_local, norm_layer)
    elif netG == 'encoder': netG = Encoder(input_nc, output_nc, ngf, n_downsample_global, norm_layer)
    else: raise('generator not implemented!')
    
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init)
    return netG

def define_D(input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1, getIntermFeat=False, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)
    
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD
    
def define_C(opt):
        from .fitting_classifier import networks
        from .fitting_classifier import options

        # Setting the model
        model = networks.DenseNetMulti(nchannels=opt.input_nc_C, drop_rate=True)
        # model.initialize(opt)

        '''substitue the initialize() function'''
        # setting the last layer
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 2)
        
        if opt.checkpoint_C: 
                model_path = opt.checkpoint_C
                model.load_state_dict(torch.load(model_path))
                print('Loaded C model @: ', model_path)
        ''''''

        return model

def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

##############################################################################
# Losses
##############################################################################

class CosineLoss1(nn.Module):
    def __init__(self):
        super(CosineLoss1, self).__init__()

    def forward(self, x, y):
        return cosine_similarity(x, y)

class CosineLoss2(nn.Module):
    def __init__(self):
        super(CosineLoss2, self).__init__()

    def forward(self, x, y):
        x_n = normalize(x)
        y_n = normalize(y)
        return 2 - (x_n).mul(y_n).mean()

class KLN01Loss(torch.nn.Module):

    def __init__(self, direction, minimize):
        super(KLN01Loss, self).__init__()
        self.minimize = minimize
        assert direction in ['pq', 'qp'], 'direction?'
        self.direction = direction

    def forward(self, samples):
        # print(samples.nelement(), samples.shape)
        # assert samples.nelement() == samples.size(1) * samples.size(0), 'wtf?'

        samples = samples.view(samples.size(0), -1); #print(samples.shape)

        self.samples_var = var(samples)
        self.samples_mean = samples.mean(1)

        samples_mean = self.samples_mean
        samples_var = self.samples_var

        if self.direction == 'pq':
            # mu_1 = 0; sigma_1 = 1
            t1 = (1 + samples_mean.pow(2)) / (2 * samples_var.pow(2))
            t2 = samples_var.log()
            
            # t1 = torch.from_numpy(np.array(t1, dtype='int32')).double()
            # t2 = torch.from_numpy(np.array(t2, dtype='int32')).double()
            
            KL = (t1 + t2 - 0.5).mean()
        else:
            # mu_2 = 0; sigma_2 = 1
            t1 = (samples_var.pow(2) + samples_mean.pow(2)) / 2
            t2 = -samples_var.log()

            # t1 = torch.from_numpy(np.array(t1, dtype='int32')).double()
            # t2 = torch.from_numpy(np.array(t2, dtype='int32')).double()

            KL = (t1 + t2 - 0.5).mean()

        if not self.minimize: KL *= -1

        return KL

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss() # nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                # print('IsList', len(input), len(input_i), input_i[-1].shape)
                # IsList 2 5 torch.Size([1, 1, 19, 19])
                target_tensor = self.get_target_tensor(pred, target_is_real)
                # print(len(input_i), pred.shape, target_tensor.shape)
                # print(pred.data.min())
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            print('IsNotList', input.shape, input[-1].shape)
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            print('net output', input[-1].data.cpu().numpy())
            print('target_tensor', target_tensor.data.cpu().numpy())
            return self.loss(input[-1], target_tensor)

class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

