# -*- coding: utf-8 -*-
# @Author: Song Dejia
# @Date:   2018-10-23 19:02:56
# @Last Modified by:   Song Dejia
# @Last Modified time: 2018-10-23 19:14:58

# this is a script to combine check_dataloader img
import os
import cv2
import numpy as np
import sys
dir_path = '/home/song/workspace/deeplab/check/check_dataloader/img'

for i in range(1, 1+len(os.listdir(dir_path))):
	o = os.path.join(dir_path, '{}_original.jpg'.format(i))
	r = os.path.join(dir_path, '{}_restore.jpg'.format(i))
	m = os.path.join(dir_path, '{}_zmask.jpg'.format(i))
	n = os.path.join(dir_path, '{}_combine.jpg'.format(i))
	img1 = cv2.imread(o)
	img2 = cv2.imread(r)
	img3 = cv2.imread(m)
	img = np.concatenate((img1, img2, img3), axis = 1)
	cv2.imwrite(n, img)
	print(i)



