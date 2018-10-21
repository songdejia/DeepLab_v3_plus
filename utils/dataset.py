# -*- coding: utf-8 -*-
# @Author: Song Dejia
# @Date:   2018-10-21 13:01:06
# @Last Modified by:   Song Dejia
# @Last Modified time: 2018-10-21 23:24:03
import sys
sys.path.append('../')
import os
import shutil
import cv2
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
    transform_list.append(FixedResize(size = (fixed_scale, fixed_scale)))
    #transform_list.append(RandomSized(fixed_scale))
    #transform_list.append(RandomRotate(rotate_prob))
    #transform_list.append(RandomHorizontalFlip())
    transform_list.append(Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    
    #transform_list.append(Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    transform_list.append(ToTensor())

    return transforms.Compose(transform_list)

def transform_for_demo(fixed_scale = 512, rotate_prob = 15):
    transform_list = []
    transform_list.append(FixedResize(size = (fixed_scale, fixed_scale)))
    transform_list.append(Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)))
    transform_list.append(ToTensor())
    return transforms.Compose(transform_list)


def transform_for_test(fixed_scale = 512):
    transform_list = []
    transform_list.append(FixedResize(size = (fixed_scale, fixed_scale)))
    transform_list.append(Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    transform_list.append(ToTensor())
    return transforms.Compose(transform_list)


def prepare_for_train_dataloader(dataroot, bs_train = 4, shuffle = True, num_workers = 2, check_dataloader = False):
    """
    use pascal 2012, dataroot contain JEPG/SEGMENTATION/so on
    """
    transform = transform_for_train(fixed_scale = 512, rotate_prob = 15)

    voc_train = VOCSegmentation(base_dir = dataroot, split = 'train', transform = transform)

    dataloader = DataLoader(voc_train, batch_size = bs_train, shuffle = False, num_workers = num_workers, drop_last = True)

    if check_dataloader:
        """
        check dataloader img
        imgs save to workspace/check/dataloader/img
        mask save to workspace/check/dataloader/mask
        
        人类；
        动物（鸟、猫、牛、狗、马、羊）；
        交通工具（飞机、自行车、船、公共汽车、小轿车、摩托车、火车）；
        室内（瓶子、椅子、餐桌、盆栽植物、沙发、电视）
        """
        transform = transform_for_demo()
        voc_train_o = VOCSegmentation(base_dir = dataroot, split = 'train', transform = transform)
        dataloader_o = DataLoader(voc_train, batch_size = bs_train, shuffle = False, num_workers = num_workers, drop_last = True)
        workspace = os.path.abspath('./')
        img_dir = os.path.join(workspace, 'check/check_dataloader/img')
        mask_dir= os.path.join(workspace, 'check/check_dataloader/mask')
        #std = np.array((0.229, 0.224, 0.225))
        #mean= np.array((0.485, 0.456, 0.406))
        std = np.array((0.5, 0.5, 0.5))
        mean= np.array((0.5, 0.5, 0.5))

        #print(img_dir)
        if os.path.exists(img_dir):
            shutil.rmtree(img_dir)

        os.makedirs(img_dir)

        idx = 0
        for index, sample_batched in enumerate(dataloader):
            inputs, labels = sample_batched['image'], sample_batched['label']
            #print('inputs',inputs.shape) #20, 3, 512, 512
            #print('labels',labels.shape) #20, 1, 512, 512
            batch = inputs.shape[0]
            for i in range(batch):
                idx += 1
                img = inputs[i].numpy().astype(np.float32)
                mask= labels[i].numpy().astype(np.float32)

                img = img.transpose((1, 2, 0))
                mask= mask.transpose((1,2, 0))
                mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

                #print(img.shape)
                img *= std
                img += mean
                img *= 255.0
                #img = np.clip(img, 0, 255)
                img = img.astype(np.int32)

                img_name = '{}_restore.jpg'.format(idx)
                img_path = os.path.join(img_dir, img_name)
                cv2.imwrite(img_path, img[:,:,::-1])
                
                mask_name= '{}_zmask.jpg'.format(idx)
                mask_path= os.path.join(img_dir, mask_name)
                cv2.imwrite(mask_path, 10*mask_rgb[:,:,::-1])

                print('idx : {:04d}'.format(idx))

        idx = 0
        for index, sample_batched in enumerate(dataloader_o):
            inputs, labels = sample_batched['image'], sample_batched['label']
            #print('inputs',inputs.shape) #20, 3, 512, 512
            #print('labels',labels.shape) #20, 1, 512, 512
            batch = inputs.shape[0]
            for i in range(batch):
                idx += 1
                
                img = inputs[i].numpy().astype(np.float32)
                mask= labels[i].numpy().astype(np.float32)
                
                img = img.transpose((1, 2, 0))
                mask= mask.transpose((1,2, 0))

                #print(img.shape)
                #img *= std
                #img += mean
                img *= 255.0
                #img = np.clip(img, 0, 255)
                img = img.astype(np.int32)

                img_name = '{}_original.jpg'.format(idx)
                img_path = os.path.join(img_dir, img_name)
                cv2.imwrite(img_path, img[:,:,::-1])
                print('idx : {:04d}'.format(idx))


        print(len(dataloader))







    return dataloader


def prepare_for_val_dataloader(dataroot, bs_val = 6, shuffle = False, num_workers = 0):
    """
    use pascal 2012
    """
    transform = transform_for_test(fixed_scale = 512)

    voc_test = VOCSegmentation(base_dir = dataroot, split = 'val', transform = transform)

    dataloader = DataLoader(voc_test, batch_size = bs_val, shuffle = shuffle, num_workers = num_workers)

    return dataloader


class VOCSegmentation(Dataset):
    """
    PascalVoc dataset
    """

    def __init__(self,
                 base_dir='./dataset',
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