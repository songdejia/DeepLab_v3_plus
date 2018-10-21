# -*- coding: utf-8 -*-
# @Author: Song Dejia
# @Date:   2018-10-21 12:18:56
# @Last Modified by:   Song Dejia
# @Last Modified time: 2018-10-21 21:15:19
import torch
import torch.nn as nn
import config
import torch.backends.cudnn as cudnn
from utils.dataset import prepare_dor_val_dataloader, prepare_for_train_dataloader
from model.deeplab import DeepLabv3_plus
from utils.util import *

def train_deeplab_v3_plus(train_dataloader, net, criterion, optimizer, epoch, global_step, running_loss_tr):
    total_loss = 0.0
    num_img_tr = len(train_dataloader)
    net.train()
    for ii, sample_batched in enumerate(train_dataloader):
        inputs, labels = sample_batched['image'], sample_batched['label']
        inputs = inputs.cuda() if config.gpu else inputs
        labels = labels.cuda() if config.gpu else labels
        global_step += inputs.shape[0]
        outputs = net(inputs)

        loss = criterion(outputs, labels, size_average=False, batch_average=True)
        running_loss_tr += loss.item()
        total_loss += loss
        avg_loss = total_loss /((ii+1)*inputs.shape[0])

        #running_loss_tr += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Train <==> Epoch:{:03d} Batch:{:03d}/{:03d} loss:{:.2f}'.format(epoch, ii+1, len(train_dataloader), loss))
    running_loss_tr /= num_img_tr
    print('Train <==> Epoch:{:03d} avg loss:{:.2f}'.format(epoch, running_loss_tr))







