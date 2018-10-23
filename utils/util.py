# -*- coding: utf-8 -*-
# @Author: Song Dejia
# @Date:   2018-10-21 14:15:52
# @Last Modified by:   Song Dejia
# @Last Modified time: 2018-10-23 19:01:24
import sys
import os
import numpy as np
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
from torch.nn import init
############################################
# parse parameters for the model
############################################
def parse_parameters():
    parser = argparse.ArgumentParser(description='PyTorch DeepLabv3_plus training')

    # parameters for dataset and network
    parser.add_argument('--dataroot', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--j', default=4, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('--gpu', default=2, type=int,
                        help='gpu num')
    parser.add_argument('--max_epoches', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--train_batchsize_per_gpu', default=14, type=int,
                        metavar='N', help='mini-batch size (default: 64)')
    parser.add_argument('--test_batchsize', default=1, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--test_epoches', default=10, type=int,
                        metavar='N', help='every test_epoches to test and save log')
    parser.add_argument('--save_weights_epoches', default=10, type=int,
                        metavar='N', help='every save_weights_epoches to save weights')

    # parameters for optimizer  
    parser.add_argument('--init_type',  default='xavier', type=str,
                        metavar='INIT',help='init net')
    parser.add_argument('--init_lr', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--decay_every_epoches', default=10, type=int,
                        metavar='N', help='every epoches to decay lr')
    parser.add_argument('--momentum', default=0.9, type=float,
                        metavar='momentum', help='momentum')
    parser.add_argument('--weight_decay', '--wd', default=1e-5, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--log_train_dir', default='log_train', type=str,
                        help='log for train')
    parser.add_argument('--log_test_dir', default='log_test', type=str,
                        help='log for test')
    parser.add_argument('--weights', default='', type=str,
                        help='weights_backup')
    parser.add_argument('--check_dataloader', default=False,type=bool)
    args = parser.parse_args()

    return args


############################################
# init weight
############################################
def init_net(net, init_type='normal'):
    init_weights(net, init_type)
    return net


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        # this will apply to each layer
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('conv')!=-1 or classname.find('Linear')!=-1):
            if init_type=='normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')#good for relu
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
    #print('initialize network with %s' % init_type)
    net.apply(init_func)


############################################
# save checkpoint and resume
############################################
def save_checkpoint(state, weights_dir = '' ):
    """[summary]
    
    [description]
    
    Arguments:
        state {[type]} -- [description] a dict describe some params
        is_best {bool} -- [description] a bool value
    
    Keyword Arguments:
        filename {str} -- [description] (default: {'checkpoint.pth.tar'})
    """
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    
    epoch = state['epoch']

    file_path = os.path.join(weights_dir, 'model-{:04d}.pth.tar'.format(int(epoch)))  
    torch.save(state, file_path)
    

#############################################
# loss function
#############################################
class AverageMeter(object):
    """
    平均值用于更新平均loss
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
 

def cross_entropy2d(logit, target, ignore_index=255, weight=None, size_average=True, batch_average=True):
    """
    logit 是网络输出 (batchsize, 21, 512, 512) 值应该为任意(没经历归一化)
    target是gt      (batchsize,  1, 512, 512) 值应该是背景为0，其他类分别为1-20，忽略为255
    return 经过h*w*batchsize平均的loss
    这里的loss相当于对每个像素点求分类交叉熵
    ignore_index 是指target中有些忽略的(非背景也非目标，是不属于数据集类别的其他物体，不计算loss) 表现为白色
    最后要注意：crossentropy是已经经过softmax，所以网络最后一层不需要处理
    https://pytorch.org/docs/stable/nn.html#torch.nn.CrossEntropyLoss
    """
    n, c, h, w = logit.size()
    # logit = logit.permute(0, 2, 3, 1)
    target = target.squeeze(1)# (batchsize, 1, 512, 512) -> (batchsize, 512, 512)
    if weight is None:
        criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, size_average=False)
    else:
        criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(weight)).float().cuda(), ignore_index=ignore_index, size_average=False)
    loss = criterion(logit, target.long())

    if size_average:
        loss /= (h * w)

    if batch_average:
        loss /= n

    return loss


def adjust_learning_rate(init_lr, optimizer, epoch, max_epoch, gamma=0.1, decay_step = 10):
    """Sets the learning rate to the initial LR decayed 0.9 every 50 epochs"""

    #lr = lr_poly(init_lr, epoch, max_epoch, power = gamma)

    lr = init_lr * (gamma ** (epoch // decay_step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr, optimizer

def lr_poly(base_lr, iter_, max_iter=100, power=0.9):
    return base_lr * ((1 - float(iter_) / max_iter) ** power)


#############################################
# evaluation and calculate mIOU
#############################################
def get_iou(pred, gt, n_classes=21):
    """
    pred 指的是每个像素所选取的通道数  [batchsize，512，512]值为通道序号：即类别
    gt   指的是target               [batchsize, 1, 512, 512]           值为类别
    对于每张图的每个类都会计算一个inter union
    返回这batchsize张图的平均iou
    """

    total_iou = 0.0
    for i in range(len(pred)):
        # 对每一个batchsize中的每张图
        # len(pred) = len(shape(batchsize, 512, 512)) = batchsize
        pred_tmp = pred[i]            #(512, 512)
        gt_tmp = gt[i].squeeze(0)     #(512, 512)

        intersect = [0] * n_classes
        union = [0] * n_classes
        for j in range(n_classes):
            """
            对于每个类别来说
            寻找预测图和背景图都为这一类的区域为positive,这里相交面积一定小于gt中类别j的面积
            正确的地方为2
            预测错了和漏预测的地方为1
            其它地方为0

            对于每批图片产生两个列表
            list1 -- intersection
            list2 -- union
            """
            match = (pred_tmp == j) + (gt_tmp == j) 

            it = torch.sum(match == 2).item()
            un = torch.sum(match > 0).item()

            intersect[j] += it
            union[j] += un

        iou = []
        for k in range(n_classes):
            if union[k] == 0:
                continue
            iou.append(intersect[k] / union[k])

        img_iou = (sum(iou) / len(iou))
        total_iou += img_iou

    return total_iou/len(pred)

def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames if filename.endswith(suffix)]

#####################################################
# 解码分割图
#####################################################
def get_cityscapes_labels():
    return np.array([
        # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]])

def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128]])


def encode_segmap(mask):
    """Encode segmentation label images as pascal classes
    Args:
        mask (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the Pascal classes are encoded as colours.
    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_pascal_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask


def decode_seg_map_sequence(label_masks, dataset='pascal'):
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask, dataset)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks

def decode_segmap(label_mask, dataset='pascal', plot=False):
    """
    将 mask 解码成class-wise rgb图
    返回 opencv 格式的 h, w, 3, 记住是np.int32
    Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    if dataset == 'pascal':
        n_classes = 21
        label_colours = get_pascal_labels()
    elif dataset == 'cityscapes':
        n_classes = 19
        label_colours = get_cityscapes_labels()
    else:
        raise NotImplementedError

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r 
    rgb[:, :, 1] = g 
    rgb[:, :, 2] = b 
    #rgb = rgb.transpose((2,0,1))
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb.astype(np.int32)

def generate_param_report(logfile, param):
    log_file = open(logfile, 'w')
    for key, val in param.items():
        log_file.write(key + ':' + str(val) + '\n')
    log_file.close()






