# -*- coding: utf-8 -*-
# @Author: Song Dejia
# @Date:   2018-10-21 13:01:06
# @Last Modified by:   Song Dejia
# @Last Modified time: 2018-10-21 14:30:20
import sys
sys.path.append('../')
import config
import os
from PIL import Image
from utils.transform import *
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def transform_for_train(fixed_scale = 512, rotate_prob = 15):
	"""
	Options:
	1.RandomCrop
	2.CenterCrop
	3.RandomHorizontalFlip
	4.Normalize
	5.ToTensor
	6.FixedResize
	7.RandomRotate
	"""
	transform_list = [] 
	transform_list.append(RandomSized(fixed_scale))
	transform_list.append(RandomRotate(rotate_prob))
	transform_list.append(RandomHorizontalFlip())
	transform_list.append(Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
	transform_list.append(ToTensor())

	return transforms.Compose(transform_list)


def transform_for_test(fixed_scale = 512):
	transform_list = []
	transform_list.append(FixedResize(size = (fixed_scale, fixed_scale)))
	transform_list.append(Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
	transform_list.append(ToTensor())

	return transforms.Compose(transform_list)


def prepare_for_train_dataloader(bs_train = 4, shuffle = True, num_workers = 2):
	"""
	use pascal 2012
	"""
	transform = transform_for_train(fixed_scale = 512, rotate_prob = 15)

	voc_train = VOCSegmentation(base_dir = config.voc2012_dataset, split = 'train', transform = transform)

	dataloader = DataLoader(voc_train, batch_size = bs_train, shuffle = shuffle, num_workers = num_workers)

	return dataloader


def prepare_dor_val_dataloader(bs_val = 6, shuffle = False, num_workers = 0):
	"""
	use pascal 2012
	"""
	transform = transform_for_test(fixed_scale = 512)

	voc_test = VOCSegmentation(base_dir = config.voc2012_dataset, split = 'val', transform = transform)

	dataloader = DataLoader(voc_test, batch_size = bs_val, shuffle = shuffle, num_workers = num_workers)

	return dataloader


class VOCSegmentation(Dataset):
    """
    PascalVoc dataset
    """

    def __init__(self,
                 base_dir=config.voc2012_dataset,
                 split='train',
                 transform=None
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'JPEGImages')
        self._cat_dir = os.path.join(self._base_dir, 'SegmentationClass')

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.transform = transform

        _splits_dir = os.path.join(self._base_dir, 'ImageSets', 'Segmentation')

        self.im_ids = []
        self.images = []
        self.categories = []

        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '.txt')), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                _image = os.path.join(self._image_dir, line + ".jpg")
                _cat = os.path.join(self._cat_dir, line + ".png")
                assert os.path.isfile(_image)
                assert os.path.isfile(_cat)
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)

        assert (len(self.images) == len(self.categories))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        _img, _target= self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def _make_img_gt_point_pair(self, index):
        # Read Image and Target
        # _img = np.array(Image.open(self.images[index]).convert('RGB')).astype(np.float32)
        # _target = np.array(Image.open(self.categories[index])).astype(np.float32)

        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.open(self.categories[index])

        return _img, _target

    def __str__(self):
        return 'VOC2012(split=' + str(self.split) + ')'