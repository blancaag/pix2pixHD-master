import argparse
import os
import torch

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
                              
        # general
        self.parser.add_argument('--data_root', type=str, default='/media/dataserver/workspace/blanca/training_datasets/classifier/training', help='training data dir')        
        # self.parser.add_argument('--data_dir', type=str, default='/media/dataserver/workspace/blanca/training_datasets/classifier/train-val', help='training data dir')
        self.parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='checkpoint files dir')
        self.parser.add_argument('--checkpoint', default=None) #'testv1_bs512_do.2', help='file name @checkpoints_dir when loading a pretrained mode')
        # 'C_net_loss_0.055400_acc_0.791500_1521033.pth'  |  testv1_bs512_do.2
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')                       
        self.parser.add_argument('--name', type=str, required=True, help='run/experiment name')
        self.parser.add_argument('--phase', default='train', help='run/experiment name')
        self.parser.add_argument('--phases', default=['train', 'val'], help='relates to the data directories names')
        self.parser.add_argument('--not_save',  action='store_true')
        
        # data loader
        self.parser.add_argument('--nc_input', type=int, default=4, help='')
        self.parser.add_argument('--nworkers', type=int, default=16, help='number of epocs') 
        self.parser.add_argument('--target_size', type=tuple, default=(224, 224), help='checkpoint files dir')
        self.parser.add_argument('--batch_size', type=int, default=16, help='if (2) 12Gb-GPU: 32 elif 6Gb-GPU: 4')
        # 32 6 GPUs; 16 4GPUs or 8 2GPUs
        self.parser.add_argument('--type', type=str, default='fromSub', help='fromSub | normal')        
        
        # model training
        self.parser.add_argument('--nepochs', type=int, default=30, help='number of epocs')
        self.parser.add_argument('--optimizer', type=str, default='adam', help='adam | SGD')
        self.parser.add_argument('--lr', type=float, default=1e-03, help='number of epocs')
        self.parser.add_argument('--lr_decay', type=int, default=15, help='epochs step size for lr decay')
        self.parser.add_argument('--beta1', type=float, default=0.9, help='Adam optimizer: beta1')
        self.parser.add_argument('--beta2', type=float, default=0.999, help='Adam optimizer: beta2')
        self.parser.add_argument('--finetune', action='store_true', help='freeze base model but not the classifier')

        self.parser.add_argument('--model', type=str, default='densenet', help='dropput value if desired to include dropout layers at the end of each densenet block')        
        self.parser.add_argument('--drop_out', type=float, default=0, help='dropput value if desired to include dropout layers at the end of each densenet block')

    def parse(self):
        if not self.initialized: self.initialize()
        self.opt = self.parser.parse_args()

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0: self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0: torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()): print('%s: %s' % (str(k), str(v)))
        print('-------------- End ---------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoint_dir, self.opt.name)
        if not os.path.exists(expr_dir): os.makedirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()): opt_file.write('%s= %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt