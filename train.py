# -*- coding: utf-8 -*-
# @Author: Song Dejia
# @Date:   2018-10-21 12:18:56
# @Last Modified by:   Song Dejia
# @Last Modified time: 2018-10-23 17:32:16
import sys
sys.path.append('../')
import torch
import torch.backends.cudnn as cudnn
from model.deeplab import DeepLabv3_plus
from utils.util import *

def train_deeplab_v3_plus(train_dataloader, net, criterion, optimizer, epoch, global_step, running_loss_tr, use_gpu = True):
    """
    global_step是指全局总共训练多少图片，这里可以有bug，不使用
    """
    losses = AverageMeter()

    total_loss = 0.0
    num_img_tr = len(train_dataloader)
    net.train()
    for ii, sample_batched in enumerate(train_dataloader):
        """
        inputs (batchsize, 3, 512, 512) --input
        predict(batchsize,21, 512, 512) --predict
        labels (batchsize, 1, 512, 512) --target
        loss 输出的loss是经过像素和bs平均化的//current avg loss in a batch
        loss -- torch.tensor // loss.item() -- normal number
        """
        inputs, labels = sample_batched['image'], sample_batched['label'] 
        inputs = inputs.cuda() if use_gpu else inputs
        labels = labels.cuda() if use_gpu else labels
        global_step += inputs.shape[0]
        outputs = net(inputs)

        loss = criterion(outputs, labels, ignore_index = 255, size_average=True, batch_average=True)
        losses.update(loss.item(), n = inputs.shape[0])

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Train <==> Epoch:{:03d} Batch:{:03d}/{:03d} *** current loss:{:.2f} *** avg loss:{:.2f}'.format(epoch, ii+1, len(train_dataloader), losses.val, losses.avg))
    






