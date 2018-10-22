# -*- coding: utf-8 -*-
# @Author: Song Dejia
# @Date:   2018-10-21 20:47:07
# @Last Modified by:   Song Dejia
# @Last Modified time: 2018-10-22 11:12:10
import torch
import torch.backends.cudnn as cudnn
import argparse
import sys
import os
from utils.util import *
from utils.dataset import *
from model.deeplab import DeepLabv3_plus
from train import train_deeplab_v3_plus
from test import evaluation


def main():
    args = parse_parameters()

    print('DeepLab_v3_plus <==> Prepare for Dataset <==> Begin')
    train_dataloader = prepare_for_train_dataloader(args.dataroot, bs_train=args.train_batchsize_per_gpu * args.gpu, shuffle=True, num_workers=args.j, check_dataloader=False)
    val_dataloader   = prepare_for_val_dataloader(args.dataroot, bs_val=args.test_batchsize, shuffle=False, num_workers=args.j) 
    print('DeepLab_v3_plus <==> Prepare for Dataset <==> Done')
    

    # network
    print('DeepLab_v3_plus <==> Prepare for Network <==> Begin')
    net = DeepLabv3_plus(nInputChannels=3, n_classes=21, os=16, pretrained=True, _print=True)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.init_lr, momentum=args.momentum, weight_decay = args.weight_decay)
    criterion = cross_entropy2d
    net = nn.DataParallel(net, device_ids=range(args.gpu)) if args.gpu else net
    net = net.cuda()
    cudnn.benchmark = True
    print('DeepLab_v3_plus <==> Initialize Network with {}<==> Begin'.format(args.init_type))
    init_weights(net, init_type=args.init_type)
    print('DeepLab_v3_plus <==> Initialize Network <==> Done')
    print('DeepLab_v3_plus <==> Resume Network checkpoint <==> Begin')
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch'] + 1
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('DeepLab_v3_plus <==> Resume Network checkpoint, next epoch:{}<==> Begin'.format(start_epoch))
    else:
        start_epoch = 0

    print('DeepLab_v3_plus <==> Prepare for Network <==> Done')


    global_step = 0
    running_loss_tr = 0.0
    for epoch in range(start_epoch, args.max_epoches):
        # train
        lr, optimizer = adjust_learning_rate(args.init_lr, optimizer, epoch, gamma = 0.9)
        
        train_deeplab_v3_plus(train_dataloader, net, criterion, optimizer, epoch, global_step, running_loss_tr)


        if epoch % 5 == 0:
            eval(val_dataloader, net, epoch)

        if epoch % 10 == 0:
            save_checkpoint({
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'optimizer' : optimizer.state_dict(),
            },
            weights_dir = args.weights)

if __name__ == '__main__':
    main()