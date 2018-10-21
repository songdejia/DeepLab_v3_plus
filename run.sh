CUDA_VISIBLE_DEVICES=0,1 python main.py \
--gpu=2 \
--j=4 \
--dataroot='/home/songyu/djsong/deeplab/dataset/VOCdevkit/VOC2012' \
--train_batchsize_per_gpu=10 \
--test_batchsize=2 \
