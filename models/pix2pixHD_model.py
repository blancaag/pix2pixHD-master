### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
import numpy as np
import torch
from torchvision import transforms
from torch.autograd import Variable

from util.image_pool import ImagePool
from .base_model import BaseModel

from data.base_dataset import resize_image_OpenCV
from . import networks_utils

def onehot_label(x):
        '''converts label to [torch.cuda.FloatTensor of size 1x2]:'''
        y_ = x.data.long()
        y = y_.view(len(y_), 1)
        y_onehot = torch.cuda.FloatTensor(len(y), 2).zero_().scatter_(1, y, 1)
        x_onehot = Variable(y_onehot)
        return x_onehot

class Pix2PixHDModel(BaseModel):
    def name(self):
        return 'Pix2PixHDModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        # if opt.resize_or_crop != 'none': # when training at full res this causes OOM
        torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        self.opt = opt

        ##### DEFINE NETWORKS ##### 
        
        # Generator network
        netG_input_nc = self.opt.input_nc # 3
        self.netG = networks_utils.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers,
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)

        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = self.opt.input_nc + opt.output_nc # 3 + 3

            self.netD = networks_utils.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid,
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)
                                          
        # Good/Bad-fit Classification network (DenseNet201)
        if self.isTrain and (opt.class_loss_G or opt.class_loss_D): 
                self.netC = networks_utils.define_C(opt)
                for param in self.netC.parameters(): param.requires_grad = opt.train_C
                self.netC.train(opt.train_C)


        print('---------- Networks initialized -------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)

        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            self.criterionGAN = networks_utils.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionFeat = torch.nn.L1Loss()
            self.criterionL1 = torch.nn.L1Loss() # adding L1 G loss
            self.criterionCos1 = networks_utils.CosineLoss1() # adding cosine G loss
            self.criterionCos2 = networks_utils.CosineLoss2() # adding cosine G loss            
            
            # classifier losses 
            self.criterionC_BCEL = torch.nn.BCEWithLogitsLoss()
            self.criterionC_BCE = torch.nn.BCELoss()
            self.criterionC_MSE = torch.nn.MSELoss()
            self.criterionFeatMap = torch.nn.L1Loss()
            
            criterion = 'param'; KL = 'qp'
            if criterion == 'param':
                # print('Using parametric criterion KL_%s' % KL)
                # KL_minimizer = losses.KLN01Loss(direction=opt.KL, minimize=True)
                # KL_maximizer = losses.KLN01Loss(direction=opt.KL, minimize=False)
            
                self.criterionKL_min = networks_utils.KLN01Loss(direction=KL, minimize=True)
                self.criterionKL_max = networks_utils.KLN01Loss(direction=KL, minimize=False)
            
            if not opt.no_vgg_loss: self.criterionVGG = networks_utils.VGGLoss(self.gpu_ids)

            # names so we can breakout loss
            self.loss_names = [
                               # 'G_cos1_z', 'G_cos2_z', 'G_cos1', 'G_cos2',
                               # 'E_KL_real', 'E_KL_fake', 'G_KL_fake',
                               ## added: G_cos1, G_cos2, G_L1, G_KL, E_KL (E_KL_real, E_KL_fake)
                               
                               'C', 
                               'G_GAN_C', 'G_GAN', 'G_GAN_Feat', 'G_VGG', 
                               'D_real', 'D_fake', 'D_Feat_real', 'D_Feat_fake'] 
            
            self.loss_weights = [
                                 # opt.lambda_G_cos1_z,
                                 # opt.lambda_G_cos2_z,
                                 # opt.lambda_G_cos1,
                                 # opt.lambda_G_cos2,
                                 # opt.lambda_E_KL_real,
                                 # opt.lambda_E_KL_fake,
                                 # opt.lambda_G_KL_fake,
                                 # opt.lambda_L1,
                                 
                                 opt.vis_C_loss,
                                 opt.lambda_G_class,
                                 1.0, 
                                 0 if opt.no_ganFeat_loss else opt.lambda_feat, 
                                 0 if opt.no_vgg_loss else opt.lambda_feat, 
                                 0.5, 
                                 0.5,
                                 opt.lambda_D_Feat,
                                 opt.lambda_D_Feat
                         ]
                                 
            print('===================== LOSSES =====================')
            [print ('%s: %.2f' %(i, j)) for i, j in zip(self.loss_names, self.loss_weights)]
            print('==================================================')
            
            ##### INITIALIZE OPTIMIZERS #####
            
            # optimizer G
            if opt.niter_fix_global > 0:
                print('------------- Only training the local enhancer network (for %d epochs) ------------' % opt.niter_fix_global)
                params_dict = dict(self.netG.named_parameters())
                params = []
                for key, value in params_dict.items():
                    if key.startswith('model' + str(opt.n_local_enhancers)):
                        params += [{'params':[value],'lr':opt.lr}]
                    else:
                        params += [{'params':[value],'lr':0.0}]
            else:
                params = list(self.netG.parameters())

            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))#, amsgrad=True)

            # optimizer D
            params = list(self.netD.parameters())
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))#, amsgrad=True)
            
            # optimizer C
            if opt.train_C:
                params = list(self.netC.parameters())
                self.optimizer_C = torch.optim.Adam(params, lr=opt.lr_C, betas=(opt.beta1, 0.999))
         
    def encode_input(self, input_image, input_mask=None, target=None, label=None, goodfit_label=None, infer=False):
                     
        input_image = Variable(input_image.data.cuda(), volatile=infer) # change volatile for requires_grad?

        # real images for training
        if target is not None: 
                target = Variable(target.data.cuda())

        if input_mask is not None: 
                input_mask = Variable(torch.cuda.FloatTensor(input_mask.data.float().unsqueeze(0).cuda()))
        
        if label is not None:
                label_onehot = onehot_label(label)
                
        if goodfit_label is not None:
                goodfit_label = Variable(torch.ones((1, 1))).cuda()
                goodfit_label_onehot = onehot_label(goodfit_label)
        
        return input_image, input_mask, target, label_onehot, goodfit_label_onehot

    def discriminate(self, input_image, real_or_fake_image, use_pool=False):
            
        input_concat = torch.cat((input_image, real_or_fake_image.detach()), dim=1)
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def forward(self, input_image, input_mask=None, target=None, label=None, infer=False):
        
        # Encode Inputs
        input_image, input_mask, target, label_onehot, goodfit_label_onehot = \
        self.encode_input(input_image, input_mask, target, label, infer)
        
        # Fake Generation -- decide if here or in the data_loader: here, I need the alpha channel for C
        if self.opt.input_nc == 4:
                assert input_mask is not None
                input_image = torch.cat((input_image, input_mask), dim=1) 
        
        fake_image = self.netG.forward(input_image)
        
        ##### CALCULATING LOSSES #####
        
        # instanciate optional losses
        loss_C = loss_G_GAN_C = loss_D_Feat_real = loss_D_Feat_fake = 0 
        # non-optional losses:
        # loss_G_GAN = loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake, 
        
        ##### GAN (G & D), L1 & FEATURE LOSSES #####
        
        # Fake Detection and Loss
        pred_fake_pool = self.discriminate(input_image, fake_image, use_pool=True)
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)

        # Real Detection and Loss
        pred_real = self.discriminate(input_image, target)
        loss_D_real = self.criterionGAN(pred_real, True)

        # GAN loss (Fake Passability Loss)
        pred_fake = self.netD.forward(torch.cat((input_image, fake_image), dim=1))
        loss_G_GAN = self.criterionGAN(pred_fake, True)

        # GAN feature matching loss
        loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i])-1):
                    loss_G_GAN_Feat += D_weights * feat_weights * self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat

        # VGG feature matching loss
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss: 
                loss_G_VGG = self.criterionVGG(fake_image, input_image) * self.opt.lambda_feat

        # # adding L1
        # loss_G_L1 = self.criterionL1(fake_image, input_image) * self.opt.lambda_L1
        
        # # defining encoder model
        # self.netGE = self.netG.model_down
        
        # # adding KL
        # loss_E_KL_real = self.criterionKL_min(self.netGE(real_image)) * self.opt.lambda_E_KL_real
        # loss_E_KL_fake = self.criterionKL_max(self.netGE(fake_image.detach())) * self.opt.lambda_E_KL_fake
        # loss_G_KL_fake = self.criterionKL_min(self.netGE(fake_image)) * self.opt.lambda_G_KL_fake;

        # adding Cosine loss
        # loss_G_cos1 = self.criterionCos1(fake_image, real_image) * self.opt.lambda_G_cos1
        # loss_G_cos2 = self.criterionCos2(fake_image, real_image) * self.opt.lambda_G_cos2
        # loss_G_cos1_z = self.criterionCos1(self.netGE(fake_image), self.netGE(real_image)) * self.opt.lambda_G_cos1_z
        # loss_G_cos2_z = self.criterionCos2(self.netGE(fake_image), self.netGE(real_image)) * self.opt.lambda_G_cos2_z 
        
        ###### EXTERNAL CLASSIFIER LOSS: TRANSFER LEARNING FROM A DENSENET PREVIOUSLY TRAINED TO DISTINGUISH GOOD & BAD FITTINGS #####
        
        # CALCULATE GENERATOR LOSS FOR GENERATING BAD IMAGES [VS. LABEL 1 (good-fitting image)]

        if self.opt.class_loss_G:
        
                def deb(x):
                        print('Debuging.. ')
                        print('Shape: ', x.shape)
                        print('Min/Max values: ', np.min(x), np.max(x))
                        return x

                transform = transforms.Compose([
                    # (**) with PIL:
                    # transforms.ToPILImage(),
                    # transforms.Resize((224, 224)),
                    # transforms.ToTensor(), [warning! ToTensor() rescale values between 0-1 and transforms to tensor]
                    transforms.Lambda(lambda x: np.transpose(x, (1, 2, 0))),
                    transforms.Lambda(lambda x: resize_image_OpenCV(x, (224, 224))), 
                    transforms.Lambda(lambda x: np.transpose(x, (2, 0, 1))),
                    transforms.Lambda(lambda x: np.expand_dims(x, axis=0)),
                    # alternative option --after resizing: 
                    # transforms.Lambda(lambda x: x * 255),
                    # transforms.ToTensor(),
                    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
        
                transform_mask = transforms.Compose([
                    # transforms.Lambda(lambda x: deb(x)),
                    transforms.Lambda(lambda x: resize_image_OpenCV(x, (224, 224))), 
                    # transforms.Lambda(lambda x: deb(x)),
                    transforms.Lambda(lambda x: np.expand_dims(x, axis=0)),
                    transforms.Lambda(lambda x: np.expand_dims(x, axis=0)),
                    # transforms.Lambda(lambda x: x / np.max(np.abs(x), axis=0)),
                    # transforms.Lambda(lambda x: 2*(x - np.max(x))/-np.ptp(x)-1),
                    # transforms.Lambda(lambda x: (x - np.mean(x)) / np.std(x))
                    # transforms.Lambda(lambda x: deb(x)),
                ])
        
                # (**) with PIL:
                # from PIL import Image
                # fake_image_resized = Variable(transform(fake_image.data.cpu().squeeze()).unsqueeze(0).cuda())
                
                
                input_image_resized = Variable(torch.cuda.FloatTensor(transform(input_image.data.cpu().squeeze().numpy())))
                fake_image_resized = Variable(torch.cuda.FloatTensor(transform(fake_image.data.cpu().squeeze().numpy())))
                             
                if self.opt.input_nc == 4:
                        assert input_mask is not None   
                        input_mask_resized = Variable(torch.cuda.FloatTensor(transform_mask(input_mask.data.cpu().squeeze().numpy())))
                                            
                        input_image_resized = torch.cat((input_image_resized, input_mask_resized), dim=1)
                        fake_image_resized = torch.cat((fake_image_resized, input_mask_resized), dim=1) 
                
                # convert label, preds to [torch.cuda.FloatTensor of size 1x2 (GPU 0)]:
                # _, preds = torch.max(classify_fake.data, 1)
                # if isinstance(self.criterionClass, torch.nn.modules.loss.BCEWithLogitsLoss):
                #     y = preds.view(len(preds), 1)
                #     y_onehot = torch.cuda.FloatTensor(len(y), 2).zero_().scatter_(1, y, 1)
                #     preds = y_onehot
                #
                # loss_C = self.criterionClass(classify_fake.detach(), label)
                # self.acc_C = torch.sum(preds == label.data) / (self.opt.batchSize * preds.shape[1])
                
                '''
                print(fake_image_resized.min().data.cpu().numpy(), \
                fake_image_resized.max().data.cpu().numpy())

                print(real_image_resized.min().data.cpu().numpy(), \
                real_image_resized.max().data.cpu().numpy())

                print(real_image_mask_resized.min().data.cpu().numpy(), \
                real_image_mask_resized.max().data.cpu().numpy())
                '''
                
                # with torch.no_grad():
                
                ## calculate "goof-fit" loss
                classify_fake = self.netC(fake_image_resized)
                classify_fake_sig = torch.nn.functional.sigmoid(classify_fake)
                
                # if classify_fake.data.cpu().numpy()[0][0] > 0:
                #         print(classify_fake.data.cpu().numpy())
                #         print(classify_fake_sig.data.cpu().numpy())
                
                loss_G_GAN_C1 = self.criterionC_BCEL(classify_fake, goodfit_label_onehot)
                loss_G_GAN_C2 = self.criterionC_BCE(classify_fake_sig, goodfit_label_onehot)
                loss_G_GAN_C3 = self.criterionC_MSE(classify_fake_sig, goodfit_label_onehot)
                
                loss_G_GAN_C = loss_G_GAN_C0
                # print('loss_GAN_C: ', loss_GAN_C0.data.cpu().numpy(), loss_GAN_C1.data.cpu().numpy(), loss_GAN_C2.data.cpu().numpy())
                
                ## calculate C classification loss --NOTICE: this should be calculated with the target image:
                classify_real = self.netC(input_image_resized) #.detach()
                classify_real_sig = torch.nn.functional.sigmoid(classify_real)
                                
                loss_C0 = self.criterionC_BCEL(classify_real, label_onehot) 
                loss_C1 = self.criterionC_BCE(classify_real_sig, label_onehot) 
                loss_C2 = self.criterionC_MSE(classify_real_sig, label_onehot) 
                
                loss_C = loss_C0 * opt.vis_C_loss
                # print('loss_C: ', loss_C0.data.cpu().numpy(), loss_C1.data.cpu().numpy(), loss_C2.data.cpu().numpy())
                
                # print('classify fake:', classify_fake.data.cpu().numpy())
                # print('classify fake sig:', classify_fake_sig.data.cpu().numpy())
                # print('classify real:', classify_real.data.cpu().numpy())
                # print('real label: ', label_onehot.data.cpu().numpy())
                # print('goodfit_label_onehot: ', goodfit_label_onehot.data.cpu().numpy())
                
                
                ## NOTICE: pondering the strength of the loss by the classification certainty of the C:
                ## UNCOMMENT BELOW
                eps = 0.000001
                # rescale or sig() loss_C in order to apply it as a weight:
                loss_C_weight = torch.nn.functional.sigmoid(loss_C)
                loss_G_GAN_C =  (loss_GAN_C * (1-loss_C_weight) + eps) * self.opt.lambda_G_class
                # loss_GAN_C = loss_GAN_C * self.opt.lambda_G_class
        
        
        ##### IMPLEMENTING ACTIVATION MAP LOSS USING PRETRAINED DENSENET C #####
        
        if self.opt.class_loss_D:
                
                # DISCRIMINATOR FEATURE AM
                def extract_D_am(output_D_list):
                        avg_out_s0 = [output_D_list[0][i].mean(1, keepdim=True) for i in range(len(output_D_list[0]))]
                        D_s0_am = avg_out_s0[4]
                        D_s0_am = torch.nn.Upsample(scale_factor=2, mode="bilinear")(D_s0_am)
                        return D_s0_am
                        
                real_D_am_s0 = extract_D_am(pred_fake_pool)
                fake_D_am_s0 = extract_D_am(pred_real)
                
                # CLASSIFIER FEATURE AM (After RELU)
                def extract_classifier_am(x, model=self.netC):
                        am = model.features(x.detach())
                        amr = torch.nn.functional.relu(am, inplace=True)
                        avg_amr = amr.mean(1, keepdim=True)
                        avg_amr = torch.nn.Upsample(scale_factor=10, mode="bilinear")(avg_amr)
                        return avg_amr
                
                real_C_am = extract_classifier_am(real_image_resized) #.detach())
                fake_C_am = extract_classifier_am(fake_image_resized) #.detach())
                
                # CLASSIFIER FEATURE WAM
                def extract_classifier_wam(x, model=self.netC):
                        am = model.features(x.detach())
                        amr = torch.nn.functional.relu(am, inplace=True)
                        amrp = torch.nn.functional.avg_pool2d(amr, kernel_size=7, stride=1).view(amr.size(0), -1)
                        out = model.classifier(amrp)
                        outsm = torch.nn.functional.sigmoid(out)
                        # weights weighted by the classification output
                        # here we multiply the output of the sigmoid for each class to the weights of the linear function (1024) for each class
                        w = torch.mm(outsm, Variable(model.classifier.weight.data))
                        # weighted class activation map before the avg. pooling
                        wam = torch.mul(am, w.unsqueeze(2).unsqueeze(3))
                        wam = wam.sum(1).unsqueeze(1)
                        wam = torch.nn.Upsample(scale_factor=10, mode="bilinear")(wam)
                        return wam
                
                real_C_wam = extract_classifier_wam(real_image_resized) # .detach())
                fake_C_wam = extract_classifier_wam(fake_image_resized) # .detach())
                
                # LOSS (L1) BETWEEN BOTH
                # loss_D_Feat_real = loss_D_Feat_fake = 0
                loss_D_Feat_real = self.criterionFeatMap(real_D_am_s0, real_C_am.detach()) * self.opt.lambda_D_Feat
                loss_D_Feat_fake = self.criterionFeatMap(fake_D_am_s0, fake_C_am.detach()) * self.opt.lambda_D_Feat
                
                # print('loss_D_Feat: ', loss_D_Feat_real, loss_D_Feat_fake, loss_D_Feat)
                
        ###### RETURN #####

        # only return images to visualize each display_freq value
        ims_to_return = [fake_image]
        if self.opt.input_nc == 4: ims_to_return += input_image_mask
        # maps:
        if self.opt.class_loss_D:
                ims_to_return += [
                        real_D_am_s0, 
                        fake_D_am_s0, 
                        real_C_am, 
                        fake_C_am,
                        real_C_wam,
                        fake_C_wam
        ]
                
        return [[
                # loss_G_cos1_z, loss_G_cos2_z, loss_G_cos1, loss_G_cos2,
                # loss_E_KL_real, loss_E_KL_fake, loss_G_KL_fake,
                # loss_G_GAN * class_weight, loss_G_GAN_Feat * class_weight, loss_G_VGG * class_weight, \   # uncomment (1)
                # loss_G_L1, 
                
                 loss_C,
                 loss_G_GAN_C, loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, 
                 loss_D_real, loss_D_fake, loss_D_Feat_real, loss_D_Feat_fake
                 
         ], [None for i in range(len(ims_to_return))] if not infer else ims_to_return]
         #, last_map, up_wcam]] #fake_image_resized_4C] # fake_image

    def inference(self, input_image):
        # Encode Inputs
        input_image = self.encode_input(Variable(input_image), infer=True)
        # Fake Generation
        fake_image = self.netG.forward(input_image)
        return fake_image

    # def sample_features(self, inst):
    #     # read precomputed feature clusters
    #     cluster_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.cluster_path)
    #     features_clustered = np.load(cluster_path).item()
    #
    #     # randomly sample from the feature clusters
    #     inst_np = inst.cpu().numpy().astype(int)
    #     feat_map = torch.cuda.FloatTensor(1, self.opt.feat_num, inst.size()[2], inst.size()[3])
    #     for i in np.unique(inst_np):
    #         label = i if i < 1000 else i//1000
    #         if label in features_clustered:
    #             feat = features_clustered[label]
    #             cluster_idx = np.random.randint(0, feat.shape[0])
    #             idx = (inst == i).nonzero()
    #             for k in range(self.opt.feat_num): feat_map[idx[:,0], idx[:,1] + k, idx[:,2], idx[:,3]] = feat[cluster_idx, k]
    #     return feat_map

    # def get_edges(self, t):
    #     # edge = torch.cuda.ByteTensor(t.size()).zero_()
    #     if self.gpu_ids == '-1': edge = torch.ByteTensor(t.size()).zero_()
    #     else: edge = torch.cuda.ByteTensor(t.size()).zero_()
    #     edge[:,:,:,1:] = edge[:,:,:,1:] | (t[:,:,:,1:] != t[:,:,:,:-1])
    #     edge[:,:,:,:-1] = edge[:,:,:,:-1] | (t[:,:,:,1:] != t[:,:,:,:-1])
    #     edge[:,:,1:,:] = edge[:,:,1:,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
    #     edge[:,:,:-1,:] = edge[:,:,:-1,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
    #     return edge.float()

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
        if opt.train_C: self.save_network(self.netC, 'C', which_epoch, self.gpu_ids)
        
    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd; print(lr, lrd)
        for param_group in self.optimizer_D.param_groups: param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups: param_group['lr'] = lr
        if self.opt.train_C: 
                for param_group in self.optimizer_C.param_groups: param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr