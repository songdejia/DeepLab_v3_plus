# -*- coding: utf-8 -*-
# @Author: Song Dejia
# @Date:   2018-10-21 20:46:42
# @Last Modified by:   Song Dejia
# @Last Modified time: 2018-10-23 16:38:13
import torch
import os
from utils.util import *

def evaluation(val_dataloader, net, epoch, use_gpu = True, save_dir ='/home/song/workspace/deeplab/tmp/log_test'):
    """
    评估函数 计算mIOU
    """
    ioues = AverageMeter()
    net.eval()
    for ii, sample_batched in enumerate(val_dataloader):
        """
        inputs -- (bs, 3, 512, 512)
        labels -- (bs, 1, 512, 512) 像素值为0-20 and 255
        outputs-- (bs, 21,512, 512) 像素值为任意
        """
        inputs, labels = sample_batched['image'], sample_batched['label']
        bs_test = inputs.shape[1]
        inputs = inputs.cuda() if use_gpu else inputs
        labels = labels.cuda() if use_gpu else labels

        outputs = net(inputs)

        predictions = torch.max(outputs, 1)[1]
        """
        outputs -- (bs, 21, 512, 512)
        val, index = torch.max(outputs, 1) 返回tuple 表示找每行的最大值
        index表示坐标 shape[1, 512, 512] 表示有512*512个位置，每个位置返回一个坐标，表示通道
        iou表示这batchsize的平均iou
        """
        iou = get_iou(predictions, labels)
        ioues.update(iou, inputs.shape[0])
        print('Test <==>Epoch: {:03} batch:{:03d}/{:03d} IOU:{:.2f} mIOU:{:.2f}'.format(epoch, ii+1, len(val_dataloader), ioues.val, ioues.avg))


    save_file = os.path.join(save_dir, 'log_test.txt')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        
    if not os.path.isfile(save_file):
        os.mknod(save_file)
    with open(save_file, 'a') as f:
        f.write('Test  <==> Epoch: {:03}  Miou:{:.2f} \n'.format(epoch, ioues.avg))

