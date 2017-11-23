# Dual-Path Convolutional Image-Text Embedding

This repository contains the code for our paper [Dual-Path Convolutional Image-Text Embedding](https://arxiv.org/abs/1711.05535). Thank you for your kindly attention. 

**The compelete code will be uploaded in two weeks. I am adding illustrations and comments to the code for using. You can check my progress as follows.**

### CheckList
- [x] Get word2vec weight

- [x] Data Preparation (Flickr30k)
- [ ] Train on Flickr30k
- [ ] Test on Flickr30k

- [ ] Data Preparation (MSCOCO)
- [ ] Train on MSCOCO
- [ ] Test on MSCOCO

- [ ] Data Preparation (CUHK-PEDES)
- [ ] Train on CUHK-PEDES
- [ ] Test on CUHK-PEDES



# Prepare Data
1. Extract wrod2vec weights. Follow the instruction in `./word2vector_matlab`;

2. Prepare the dataset. Follow the instruction in `./dataset`. You can choose one dataset to run.
Three datasets need different prepocessing. I write the instruction for [Flickr30k](https://github.com/layumi/Image-Text-Embedding/tree/master/data/Flickr30k), MSCOCO and CUHKPEDES.

3. Download the model pre-trained on ImageNet. And put the model into './data'.
```
(bash) wget http://www.vlfeat.org/matconvnet/models/imagenet-resnet-50-dag.mat
```
Alternatively, you may try [VGG16](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-16.mat) or [VGG19](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat). 

# Train
1. Compile Matconvnet
**(Note that I have included my Matconvnet in this repo, so you do not need to download it again. I have changed some codes comparing with the original version. For example, one of the difference is in `/matlab/+dagnn/@DagNN/initParams.m`. If one layer has params, I will not initialize it again, especially for pretrained model.)**

You just need to uncomment and modify some lines in `gpu_compile.m` and run it in Matlab. Try it~
(The code does not support cudnn 6.0. You may just turn off the Enablecudnn or try cudnn5.1)
```
(matlab) gpu_compile
```
If you fail in compilation, you may refer to http://www.vlfeat.org/matconvnet/install/

2. Prepocessing may make you lose your patience. But let's just start trainning!!

# Test
## Test
Select one model and have fun! Run `test/extract_pic_feature_word2_plus_52.m` and to extract the image features from base branch and alignment brach. Note that you need to change the model path in the code. 

If you train on CUHK-PEDES, use 'test_CUHK' to evaluate.

If you train on MSCOCO, use 'test_COCO' to evaluate.
