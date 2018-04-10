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

def on_hot_tensor(x):
    x_ = preds.view(len(x), 1)
    if use_gpu: 
        x_onehot = torch.cuda.FloatTensor(len(x_), 2).zero_().scatter_(1, x_, 1)
    else: x_onehot = torch.FloatTensor(len(x_), 2).zero_().scatter_(1, x_, 1)     
    return x_onehot  

def train_model(model, dataloaders, criterion, opt, datasets_sizes):
    
    for param in model.parameters(): 
            param.requires_grad = True
    
    if opt.finetune: 
            for param in model.parameters(): param.requires_grad = False
            for param in model.module.classifier.parameters(): param.requires_grad = True
          
    train_hist = {phase: {'loss': [], 'acc': []} for phase in opt.phases}
    
    since = time.time()

    best_model_wts = model.state_dict()
    best_loss = 1.0
    best_loss_acc = 0.0
    # best_acc = 0.0

    for epoch in range(opt.nepochs):
        print('\nEpoch {}/{}'.format(epoch, opt.nepochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in opt.phases:
            if phase == 'train':
                model.module.scheduler.step()
                print('lr: ', model.module.scheduler.get_lr()[0])
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            # print("We are in epoch %d phase %s" %(epoch, phase))
            
            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data

                # ON_HOT LABELS IF CRITERION REQUIRES IT
                # print (labels, type(labels))
                # print(criterion)
                if isinstance(criterion, torch.nn.modules.loss.BCEWithLogitsLoss):
                        y = labels.view(len(labels), 1).long()
                        y_onehot = torch.FloatTensor(len(y), 2).zero_().scatter_(1, y, 1)
                        labels = y_onehot
                        
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                model.module.optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                
                if isinstance(criterion, torch.nn.modules.loss.BCEWithLogitsLoss):
                    y = preds.view(len(preds), 1)
                    if use_gpu: y_onehot = torch.cuda.FloatTensor(len(y), 2).zero_().scatter_(1, y, 1)
                    else: y_onehot = torch.FloatTensor(len(y), 2).zero_().scatter_(1, y, 1)     
                    preds = y_onehot
                
                # print(preds, labels)
                loss = criterion(outputs, labels)
                # acc = torch.sum(preds == labels.data).item() / (opt.batch_size * preds.shape[1])
                acc = torch.sum(preds == labels.data) / (opt.batch_size * preds.shape[1])

                # print("Batch loss/accuracy", loss, acc)
                
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    model.module.optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data) / preds.shape[1]
                
                train_hist[phase]['loss'].append(loss.data[0])
                train_hist[phase]['acc'].append(acc)
                
            epoch_loss = running_loss / datasets_sizes[phase]
            epoch_acc = running_corrects / datasets_sizes[phase]
            
            print('{} -- Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            # if phase == 'val' and epoch_acc > best_acc:
                # best_acc = epoch_acc
                # best_acc_loss = epoch_loss
                # best_model_wts = model.state_dict()
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_loss_acc = epoch_acc
                best_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    print('Best val acc.: {:4f}'.format(best_loss_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    if not opt.not_save:
            print('Saving...')
            output_dir = os.path.join(opt.checkpoint_dir, opt.name)
            filename = '%s_net_loss_%f_acc_%f_%d.pth' % ('C', best_loss, best_loss_acc, int(time.time()))
            output_path = os.path.join(output_dir, filename)
            
            # print(model)
            # works: torch.save(model.module.state_dict(), output_path)

            def try_saving(model, output_path=output_path):
                success = 0
                while success==0:
                    try:
                        torch.save(model.module.cpu().state_dict(), output_path)
                        success=1
                        print(success)
                    except: 
                        torch.save(model.module.cpu().state_dict(), output_path)
            
            try: try_saving(model, output_path)
            except: try_saving(model, output_path)
                
            # torch.save(model.cpu().state_dict(), output_path)
            # torch.save(model.module.state_dict(), output_path + '_')            
            # model.module.save_state_dict(output_path + '_v2')
            # model.save_state_dict(output_path + '_')
                        
            # def save_network(self, network, network_label, epoch_label, gpu_ids):
            #     save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
            #     save_path = os.path.join(self.save_dir, save_filename)
            #     torch.save(network.cpu().state_dict(), save_path)
            #     if len(gpu_ids) and torch.cuda.is_available(): network.cuda()
            
            if torch.cuda.is_available(): model.cuda()
    
            # save history of losses to the disk
            file_name = os.path.join(output_dir, filename + '_losses.txt')
            with open(file_name, 'wt') as opt_file:
                for k, v in train_hist.items(): 
                        opt_file.write('%s: %s\n' % (str(k), str(v)))
    
    # torch.save(model.state_dict(),
    #            os.path.join(output_dir,
    #                 'model_best_c%d_%f_%f_%d.pth.tar' %(opt.nc_input, best_loss, best_loss_acc, int(time.time()/1000))))
    # torch.save(model,
    #            os.path.join(output_dir,
    #                'model_best_c%d_%f_%f_%d.pt' %(opt.nc_input, best_loss, best_loss_acc, int(time.time()/1000))))

    return model