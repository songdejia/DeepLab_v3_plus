# -*- coding: utf-8 -*-
# @Author: Song Dejia
# @Date:   2018-10-21 12:18:56
# @Last Modified by:   Song Dejia
# @Last Modified time: 2018-10-21 15:53:44
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

def eval(val_dataloader, net, epoch):
    total_iou = .0
    net.eval()
    for ii, sample_batched in enumerate(val_dataloader):
        inputs, labels = sample_batched['image'], sample_batched['label']
        inputs = inputs.cuda() if config.gpu else inputs
        labels = labels.cuda() if config.gpu else labels

        outputs = net(inputs)

        predictions = torch.max(outputs, 1)[1]
        iou = get_iou(predictions, labels)
        total_iou += iou
        print('Test <==> batch:{:03d}/{:03d} iou:{:.2f}'.format(ii+1, len(val_dataloader), iou))
    miou = total_iou / (ii * bs_val + inputs.data.shape[0])
    print('Test  <==> Epoch: {:03} total_iou:{:.2f} miou:{:.2f}'.format(epoch, total_iou, miou))





def main():
    

    # dataset
    bs_train = 24
    bs_val = 2
    max_epoches = 1000
    init_lr = 0.001
    momentum = 0.9
    weight_decay = 5e-4


    train_dataloader = prepare_for_train_dataloader(bs_train = bs_train, shuffle = True, num_workers = 2)
    val_dataloader   = prepare_dor_val_dataloader(bs_val = bs_val, shuffle = False, num_workers = 2) 

    # network
    net = DeepLabv3_plus(nInputChannels=3, n_classes=21, os=16, pretrained=True, _print=True)
    net = nn.DataParallel(net, device_ids=config.gpu_list) if config.gpu else net
    net = net.cuda()
    cudnn.benchmark = True
    init_weights(net, init_type='xavier')


    optimizer = torch.optim.SGD(net.parameters(), lr=init_lr, momentum=momentum, weight_decay = weight_decay)
    criterion = cross_entropy2d

    global_step = 0
    running_loss_tr = 0.0
    for epoch in range(max_epoches):
        # train
        lr, optimizer = adjust_learning_rate(init_lr, optimizer, epoch, gamma = 0.9)
        
        train_deeplab_v3_plus(train_dataloader, net, criterion, optimizer, epoch, global_step, running_loss_tr)


        if epoch % 10 == 0:
        	eval(val_dataloader, net, epoch)

if __name__ == '__main__':
    main()