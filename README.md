# DeepLab_V3_plus : a model about semantic segmentation
This is a simple pytorch re-implementation of Google [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/pdf/1802.02611.pdf).

### Introduction:
This work still need to be updated.
The features are summarized blow:
+ Use ResNet101 as base Network.
+ Use only Pascal Voc 2012 for base dataset.


### We have finished:
+ Dataloader for [Pascal Voc 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/).
+ You can check your dataloader error in 'path/to/workspace/check/check_dataloader/img'
  You can check three part:
  + 1.original image
  + 2.restore image from torch-tensor(transformed) back to numpy ndarray
  + 3.zmask loaded by dataloader
  e.g.
  
  original image
  <div align=center><img width="700" height="700" src="https://github.com/songdejia/deeplab_v3_plus/blob/master/screenshot/original.png"/></div>

  restore image from tensor to numpy
  <div align=center><img width="700" height="700" src="https://github.com/songdejia/deeplab_v3_plus/blob/master/screenshot/restore.png"/></div>

  mask
  <div align=center><img width="700" height="700" src="https://github.com/songdejia/deeplab_v3_plus/blob/master/screenshot/mask.png"/></div>
