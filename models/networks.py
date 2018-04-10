### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np

##############################################################################
# Generator
##############################################################################

class LocalEnhancer(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9,
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers

        ###### global generator model #####
        ngf_global = ngf * (2**n_local_enhancers)
        model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global, norm_layer).model
        model_global = [model_global[i] for i in range(len(model_global)-3)] # get rid of final convolution layers
        self.model = nn.Sequential(*model_global)

        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers+1):
            ### downsample
            ngf_global = ngf * (2**(n_local_enhancers-n))
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0),
                                norm_layer(ngf_global), nn.ReLU(True),
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1),
                                norm_layer(ngf_global * 2), nn.ReLU(True)]
            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

            ### upsample
            model_upsample += [nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1),
                               norm_layer(ngf_global), nn.ReLU(True)]

            ### final convolution
            if n == n_local_enhancers:
                model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]

            setattr(self, 'model'+str(n)+'_1', nn.Sequential(*model_downsample))
            setattr(self, 'model'+str(n)+'_2', nn.Sequential(*model_upsample))

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input):
        ### create input pyramid
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        ### output at coarest level
        output_prev = self.model(input_downsampled[-1])
        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers+1):
            model_downsample = getattr(self, 'model'+str(n_local_enhancers)+'_1')
            model_upsample = getattr(self, 'model'+str(n_local_enhancers)+'_2')
            input_i = input_downsampled[self.n_local_enhancers-n_local_enhancers]
            output_prev = model_upsample(model_downsample(input_i) + output_prev)
        return output_prev

class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect'):
        assert(n_blocks >= 0)
        super(GlobalGenerator, self).__init__()
        activation = nn.ReLU(True)
        
        # model_down_1 = [
        #         # nn.ReflectionPad2d(3),
        #         nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0)
        #
        # ]
        model_down = [
                nn.ReflectionPad2d(3),
                nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                norm_layer(ngf), 
                activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model_down += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                           norm_layer(ngf * mult * 2), activation]
            
            # self.model_down_1 = nn.Sequential(*model_down_1)
            self.model_down = nn.Sequential(*model_down)
        
        ### resnet blocks
        mult = 2**n_downsampling
        model_res = []
        for i in range(n_blocks):
            model_res += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
            
            self.model_res = nn.Sequential(*model_res)
            
        ### upsample
        model_up = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model_up += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                         norm_layer(int(ngf * mult / 2)), activation]
        model_up += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        
        self.model = nn.Sequential(*model_up)
        # self.model = self.model_up
        
        # model_enc, model_dec = model_down, model_res + model_up
        # self.model_enc, self.model_dec = nn.Sequential(*model_enc), nn.Sequential(*model_dec)

    def forward(self, input):
        txtmap_embedding = 0
        ### ### CONCAT WITH EMBEDDED TEXTURE MAP ### ###    
        # print(input.data.shape);
        # torch.Size([1, 3, 256, 256])
        # torch.Size([1, 4, 256, 256])
        # torch.Size([1, 4, 262, 262])
        # out_down_1 = self.model_down_1(input);
        # print(out_down_1.shape)
        out_down = self.model_down(input);
        # print(out_down.shape)
        # print(out_down.data.shape) # torch.Size([1, 1024, 16, 16])
        # out_concat = torch.cat(out_down, texture_embedding); print(out_concat.data[0].shape)
        out_sum = out_down + txtmap_embedding; # print(out_sum.data.shape) # torch.Size([1, 1024, 16, 16])
        out_res = self.model_res(out_sum); # print(out_res.data.shape) # torch.Size([1, 1024, 16, 16])
        out_up = self.model(out_res); # print(out_up.data.shape) # torch.Size([1, 3, 256, 256])
        return out_up

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class Encoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=4, norm_layer=nn.BatchNorm2d):
        super(Encoder, self).__init__()
        self.output_nc = output_nc

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                 norm_layer(ngf), nn.ReLU(True)]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), nn.ReLU(True)]

        ### upsample
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input, inst):
        outputs = self.model(input)

        # instance-wise average pooling
        outputs_mean = outputs.clone()
        inst_list = np.unique(inst.cpu().numpy().astype(int))
        for i in inst_list:
            indices = (inst == i).nonzero() # n x 4
            for j in range(self.output_nc):
                output_ins = outputs[indices[:,0], indices[:,1] + j, indices[:,2], indices[:,3]]
                mean_feat = torch.mean(output_ins).expand_as(output_ins)
                outputs_mean[indices[:,0], indices[:,1] + j, indices[:,2], indices[:,3]] = mean_feat
        return outputs_mean

##############################################################################
# Discriminator
##############################################################################

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers+2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else: setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        
        # self.fc = nn.Linear(d*2, 1)
        # self.avgpool_D1 = nn.AvgPool2d(35, stride=1)
        # self.avgpool_D2 = nn.AvgPool2d(35, stride=1)
        
    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)): 
                    result.append(model[i](result[-1]))
            return result[1:]
        else: return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            # select intermediate layers of each discriminator:
            if self.getIntermFeat: model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in range(self.n_layers + 2)]
            else: model = getattr(self, 'layer' + str(num_D - 1 - i))
            
            # perform forward pass on that D --> output: list of outputs by layer (5 layers/features considered)
            result.append(self.singleD_forward(model, input_downsampled))
            
            # downsample the input before going to next D
            if i != (num_D - 1): input_downsampled = self.downsample(input_downsampled)
        return result
        
    # def forward_cam():
    #     x1= self.avgpool(x)
    #     # print("After poolx ={}".format(x.shape))
    #     x1 = x1.view(x1.size(0), -1)
    #     outsm = F.sigmoid(self.fc(x1))
    #     w = torch.mm(outsm, Variable(self.fc.weight.data))
    #     cam = torch.mul(x, w.unsqueeze(2).unsqueeze(3))
    #     cam = cam.sum(1).unsqueeze(1)
    #     # print("OK")
    #     #print("outputCAM size is {}".format(self.upsample(cam).shape))
    #     return outsm, self.upsample(cam)   
                  
            
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw), norm_layer(nf), nn.LeakyReLU(0.2, True)]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw), norm_layer(nf), nn.LeakyReLU(0.2, True)]]
        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid: sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)): sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n)); print(model)
                res.append(model(res[-1]))
            return res[1:]
        else: return self.model(input)

##############################################################################
# Features
##############################################################################

from torchvision import models
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

##############################################################################
# Classifier
##############################################################################

# from .fitting_classifier import networks
# from .fitting_classifier import options
#
# class ClassDenseNet(): #networks.DenseNetMulti):
#     def name(self): return 'ClassDenseNet'
#
#     def __init__(self, opt):
#
#         # class_opt = options.Options().parse()
#         use_gpu = torch.cuda.is_available()
#
#         model = networks.DenseNetMulti(nchannels=opt.class_nc)
#         model.initialize(opt)
#         if use_gpu: model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count()))).cuda()
#
#         pretrained_1 = 'fitting_classifier/checkpoints_densenet_test/test_clean/model_best_c4_0.742152_0.018293_1519316.pt'
#         pretrained_2 = 'fitting_classifier/checkpoints_densenet_test/test_clean/model_best_c4_0.829596_0.023838_1519321.pt'
#         pretrained_3 = 'fitting_classifier/checkpoints_densenet_test/test_clean/model_best_c4_0.849776_0.022063_1519323.pt'
#
#         pretrained = '/blanca/project/wip/pix2pixHDX_class-master/models/fitting_classifier/checkpoints_densenet_test/test_clean/model_best_c4_0.849776_0.022063_1519323.pt'
#         for param in model.parameters(): param.requires_grad = False
#
#         # uncomment below if  want to add pretrained
#         checkpoint = pretrained
#         if checkpoint: model.load_state_dict(torch.load(checkpoint + 'h.tar'))


    
