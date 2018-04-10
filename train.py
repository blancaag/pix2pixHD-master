### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import time
from collections import OrderedDict
from options.train_options import TrainOptions
from data.custom_data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util.visdom_visualizer import VisdomVisualizer
import os
import numpy as np
import torch
from torch.autograd import Variable

opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
else:
    start_epoch, epoch_iter = 1, 0

if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt)
visdom_visualizer = VisdomVisualizer(opt)

total_steps = (start_epoch-1) * dataset_size + epoch_iter
class_acc_pEpoch = []
for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch: epoch_iter = epoch_iter % dataset_size
    class_acc_sum = 0
    for i, data in enumerate(dataset, start=epoch_iter):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        # whether to collect output images
        save_fake = total_steps % opt.display_freq == 0

        ############## Forward Pass ######################
        from collections import OrderedDict
        
        ims_returning_dict = OrderedDict([
                ('fake_image', None),
                ('input_image_mask', None),
                ('real_D_am_s0', None), 
                ('fake_D_am_s0', None), 
                ('real_C_am', None), 
                ('fake_C_am', None),
                ('real_C_wam', None),
                ('fake_C_wam', None)
                ])
        
        ims_returning = [ims_returning_dict['fake_image']]
        if opt.input_nc == 4: 
                ims_returning += ims_returning_dict['input_image_mask'], 
        if opt.class_loss_D: 
                for k in [
                        'real_D_am_s0', 
                        'fake_D_am_s0', 
                        'real_C_am', 
                        'fake_C_am',
                        'real_C_wam',
                        'fake_C_wam']:
                        ims_returning += ims_returning_dict[k] 
        
        # ims_returning = [ims_returning_dict[k] for k in in ims_returning_dict.items()]
        
        losses, ims_returning = model(Variable(data['input_image']), 
                                  Variable(data['input_mask']),
                                  Variable(data['target']), 
                                  Variable(data['label']), infer=save_fake)
        
        fake_image = ims_returning[0]
        
        # if opt.class_with_C: class_acc_sum += model.module.acc_C
        
        # sum per device losses
        losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
        loss_dict = dict(zip(model.module.loss_names, losses))

        # calculate final loss scalar
        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5 + loss_dict['D_Feat_real'] + loss_dict['D_Feat_fake'] 
        
        loss_G = loss_dict['G_GAN'] + loss_dict['G_GAN_Feat'] + loss_dict['G_VGG'] + loss_dict['G_GAN_C']
        
        loss_C = loss_dict['C']
                
        ############### Backward Pass ####################
        # update generator weights
        model.module.optimizer_G.zero_grad()
        loss_G.backward(retain_graph=True)
        model.module.optimizer_G.step()

        # update discriminator weights
        model.module.optimizer_D.zero_grad()
        loss_D.backward()#retain_graph=True)
        model.module.optimizer_D.step()

        #call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
        
        # update classifier weights
        if opt.train_C:
                model.module.optimizer_C.zero_grad()
                loss_C.backward()
                model.module.optimizer_C.step()
        
        ############## Display results and errors ##########
        ### print out errors
        if total_steps % opt.print_freq == 0:
            errors = {k: v.data[0] if not isinstance(v, int) else v for k, v in loss_dict.items()}
            t = (time.time() - iter_start_time) / opt.batchSize
            # visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            # visualizer.plot_current_errors(errors, total_steps)
            visdom_visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            visdom_visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        ### display output images
        if save_fake:
            visuals = OrderedDict([('input_image', util.tensor2label(data['input_image'][0], opt.input_nc)),
                                   ('synthesized_image', util.tensor2im(fake_image.data[0])),
                                   ('target_image', util.tensor2im(data['target'][0])),
                                   # ('input_mask', util.tensor2im(input_mask.data[0])),
                                   # ('real_D_am_s0', util.tensor2im(real_D_am_s0.data[0])),
                                   # ('fake_D_am_s0', util.tensor2im(fake_D_am_s0.data[0])),
                                   # ('real_C_am', util.tensor2im(real_C_am.data[0])),
                                   # ('fake_C_am', util.tensor2im(fake_C_am.data[0])),
                                   # ('real_C_wam', util.tensor2im(real_C_wam.data[0])),
                                   # ('fake_C_wam', util.tensor2im(fake_C_wam.data[0]))    
                           ])
                                         
            # visualizer.display_current_results(visuals, epoch, total_steps)
            visdom_visualizer.display_current_results(visuals, epoch, save_fake)

        ### save latest model
        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.module.save('latest')
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

    # if opt.class_with_C:
    #         class_acc_epoch = class_acc_sum / len(dataset)
    #         class_acc_pEpoch.append(class_acc_epoch)
    
    # end of epoch
    iter_end_time = time.time()
    for param_group in model.module.optimizer_G.param_groups: 
            current_lr = param_group['lr']
    print('End of epoch %d / %d \t Time Taken: %d sec \t LR: %f' # .  |   Class. accuracy %f' \
    % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time, current_lr))#, class_acc_epoch))
    
    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        model.module.save('latest')
        model.module.save(epoch)
        np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

    ### instead of only training the local enhancer, train the entire network after certain iterations
    if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global): model.module.update_fixed_params()

    ### linearly decay learning rate after certain iterations
    if epoch > opt.niter: model.module.update_learning_rate()

# print(class_acc_pEpoch)