# -*- coding: utf-8 -*-
# @Author: Song Dejia
# @Date:   2018-10-21 13:03:55
# @Last Modified by:   Song Dejia
# @Last Modified time: 2018-10-21 14:56:40
import os

gpu = True
gpu_list = [0,1]

workspace = os.path.abspath('./')
voc2012_dataset = os.path.join(workspace, 'dataset/VOCdevkit/VOC2012')	