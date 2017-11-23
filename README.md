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
* For Flickr30k, run ``
Run ``

2. 

3. 

# Test
Select one model and have fun! Run `test/extract_pic_feature_word2_plus_52.m` and to extract the image features from base branch and alignment brach. Note that you need to change the model path in the code. 

If you train on CUHK-PEDES, use 'test_CUHK' to evaluate.

If you train on MSCOCO, use 'test_COCO' to evaluate.
