# -*- coding: utf-8 -*-
# @Author: Song Dejia
# @Date:   2018-10-21 20:46:42
# @Last Modified by:   Song Dejia
# @Last Modified time: 2018-10-22 11:12:07
import torch

def evaluation(val_dataloader, net, epoch):
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
    miou = total_iou / (ii * 2 + inputs.data.shape[0])
    print('Test  <==> Epoch: {:03} total_iou:{:.2f} miou:{:.2f}'.format(epoch, total_iou, miou))