### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self) # call to a class function withough instantiating the class first?
        
        # for displays
        self.parser.add_argument('--display_freq', type=int, default=10, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=10000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')        
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')
        ## visdom:
        self.parser.add_argument('--display_single_pane_ncols', type=int, default=0, help='if positive, display all images in a single visdom web panel with certain number of images per row.')

        # for training
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--load_pretrain', type=str, default='', help='load the pretrained model from the specified location')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--niter', type=int, default=50, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=0, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')

        # for discriminators        
        self.parser.add_argument('--num_D', type=int, default=2, help='number of discriminators to use')
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')    
        self.parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')                
        self.parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
        self.parser.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')        
        self.parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--pool_size', type=int, default=0, help='the size of image buffer that stores previously generated images')
        
        self.parser.add_argument('--lambda_D_Feat', type=int, default=0, help='weight for the classsification loss')        
        
        # for the generator:
        self.parser.add_argument('--lambda_L1', type=float, default=0, help='weight for the L1 loss')
        self.parser.add_argument('--lambda_G_cos1', type=float, default=0, help='weight for the L1 loss')
        self.parser.add_argument('--lambda_G_cos2', type=float, default=0, help='weight for the L1 loss')
        self.parser.add_argument('--lambda_G_cos1_z', type=float, default=0, help='weight for the L1 loss')
        self.parser.add_argument('--lambda_G_cos2_z', type=float, default=0, help='weight for the L1 loss')
        self.parser.add_argument('--lambda_G_KL_fake', type=float, default=0, help='weight for the KL loss')
        self.parser.add_argument('--lambda_E_KL_real', type=float, default=0, help='weight for the KL_real loss')
        self.parser.add_argument('--lambda_E_KL_fake', type=float, default=0, help='weight for the KL_fake loss')
      
        self.parser.add_argument('--lambda_G_class', type=int, default=0, help='weight for the classsification loss')
        self.parser.add_argument('--lambda_G_class_weight', type=int, default=1, help='weight for the classsification loss')
                        
        # for the classifier
        self.parser.add_argument('--vis_C_loss', type=int, default=0, help='(1: True) Visualize classifier loss on visdom | screen')  
        self.parser.add_argument('--class_loss_D', type=bool, default=False, help='use pre-trained DenseNet classifier to classify the good or bad fitting of the image')
        self.parser.add_argument('--class_loss_G', type=bool, default=False, help='use pre-trained DenseNet classifier to classify the good or bad fitting of the image')
        self.parser.add_argument('--input_nc_C', type=int, default=3, help='input channels for the classifier network')
        
        self.parser.add_argument('--n_layers_C', type=int, default=3, help='number of layers for the classifier network')
        self.parser.add_argument('--train_C', action='store_true', help='if freezing or training the classifier network')
        self.parser.add_argument('--lr_C', type=float, default=0.00002, help='initial learning rate for adam')
        self.parser.add_argument('--checkpoint_C', \
                                 default='/media/dataserver/workspace/blanca/project/wip/pix2pixHDX-class-master/'\
                                 'models/fitting_classifier/checkpoints/testv1_bs512_do.2/C_net_loss_0.001614_acc_0.794788_1522613607.pth',
                                help='full path of the pretrained dense net xc model')
         
        self.isTrain = True