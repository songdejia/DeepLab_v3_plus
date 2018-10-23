# DeepLab_V3_plus : a model about semantic segmentation
This is a simple pytorch re-implementation of Google [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/pdf/1802.02611.pdf).

### Introduction:
This work still need to be updated.
The features are summarized blow:
+ Use ResNet101 as base Network. Xception will be updated soon.
+ Use only VOC2012 for base dataset. Other dataset will be updated soon.


### We have finished:
+ Version avaliable for [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/).
+ You can check your dataloader error in 'path/to/workspace/check/check_dataloader/img'.
  You will see three part:
  + 1.original image which we load directly from image path.
  + 2.restore image from torch-tensor(transformed from numpy) back to numpy ndarray.
  + 3.mask loaded by dataloader.
  Following images show original image, augumentation image and mask target from left to right 
  <div align=left><img width="900" height="300" src="https://github.com/songdejia/DeepLab_v3_plus_pytorch/blob/master/screenshot/combine1.jpg"/></div>
  <div align=left><img width="900" height="300" src="https://github.com/songdejia/DeepLab_v3_plus_pytorch/blob/master/screenshot/combine2.jpg"/></div>
  <div align=left><img width="900" height="300" src="https://github.com/songdejia/DeepLab_v3_plus_pytorch/blob/master/screenshot/combine3.jpg"/></div>

+ Network architecture.
  <div align=left><img width="800" height="450" src="https://github.com/songdejia/DeepLab_v3_plus_pytorch/blob/master/screenshot/network.jpg"/></div>
  
+ Evaluation mIOU on PASCAL VOC2012 valset every epoches. 
+ Pretrained model on PASCAL VOC2012 [BaiduYun Link](https://pan.baidu.com/s/1g75wjmnUkIDCml2S1IeDgA),
  which is trained on 2 Tesla P100 for 100 epoches with config shown in code. 

### Usage:
+ Download dataset and unzip
```
ln -s VOCdevkit path/to/deeplab_v3_plus/dataset
```
+ Pretrained model is avaliable [BaiduYun Link](https://pan.baidu.com/s/1g75wjmnUkIDCml2S1IeDgA)
+ Finally, run the model.
+ Check you GPU resources and modify your run.sh. 
```
sh run.sh
```

### Future:
+ Xception as Network Baseline.
+ Pretrained model on COCO, JFT.
+ Depthwise separable convolution.
+ Support dataset for Cityscapes.
+ Visualization for test result and gt.


  
