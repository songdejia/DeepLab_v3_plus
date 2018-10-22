# -*- coding: utf-8 -*-
# @Author: Song Dejia
# @Date:   2018-10-21 20:47:07
# @Last Modified by:   Song Dejia
# @Last Modified time: 2018-10-22 10:09:54
import torch
import argparse
import sys
import os
from utils.util import *
from utils.dataset import *


def main():
    args = parse_parameters()

    print('DeepLab_v3_plus <==> Prepare for Dataset <==> Begin')
    train_dataloader = prepare_for_train_dataloader(args.dataroot, bs_train=args.train_batchsize_per_gpu * args.gpu, shuffle=True, num_workers=args.j, check_dataloader=True)
    val_dataloader   = prepare_for_val_dataloader(dataroot, bs_val=args.test_batchsize, shuffle=False, num_workers=args.j, check_dataloader=True) 
    print('DeepLab_v3_plus <==> Prepare for Dataset <==> Done')
    

    # network
    print('DeepLab_v3_plus <==> Prepare for Network <==> Begin')
    net = DeepLabv3_plus(nInputChannels=3, n_classes=21, os=16, pretrained=True, _print=True)
    optimizer = torch.optim.SGD(net.parameters(), lr=init_lr, momentum=momentum, weight_decay = weight_decay)
    criterion = cross_entropy2d
    net = nn.DataParallel(net, device_ids=config.gpu_list) if config.gpu else net
    net = net.cuda()
    cudnn.benchmark = True
    print('DeepLab_v3_plus <==> Initialize Network with {}<==> Begin'.format(init_type))
    init_weights(net, init_type='xavier')
    print('DeepLab_v3_plus <==> Initialize Network <==> Done')
    print('DeepLab_v3_plus <==> Resume Network checkpoint <==> Begin')
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch'] + 1
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
    print('DeepLab_v3_plus <==> Resume Network checkpoint, next epoch:{}<==> Begin'.format(start_epoch))

    print('DeepLab_v3_plus <==> Prepare for Network <==> Done')
    sys.exit(0)


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