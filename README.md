# Dual-Path Convolutional Image-Text Embedding

This repository contains the code for our paper [Dual-Path Convolutional Image-Text Embedding](https://arxiv.org/abs/1711.05535). Thank you for your kindly attention. 

![](http://zdzheng.xyz/images/fulls/ConvVSE.jpg)

**What's New**: We updated the paper to the second version, adding more illustration about the mechanism of the proposed instance loss.

# Prepocess Datasets
1. Extract wrod2vec weights. Follow the instruction in `./word2vector_matlab`;

2. Prepocess the dataset. Follow the instruction in `./dataset`. You can choose one dataset to run.
Three datasets need different prepocessing. I write the instruction for [Flickr30k](https://github.com/layumi/Image-Text-Embedding/tree/master/dataset/Flickr30k-prepare), [MSCOCO](https://github.com/layumi/Image-Text-Embedding/tree/master/dataset/MSCOCO-prepare) and [CUHK-PEDES](https://github.com/layumi/Image-Text-Embedding/tree/master/dataset/CUHK-PEDES-prepare).

3. Download the model pre-trained on ImageNet. And put the model into './data'.
```
(bash) wget http://www.vlfeat.org/matconvnet/models/imagenet-resnet-50-dag.mat
```
Alternatively, you may try [VGG16](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-16.mat) or [VGG19](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat). 

# Trained Model
You may download the three trained models from [GoogleDrive](https://drive.google.com/open?id=1QxIdJd3oQIJSVVlAxaIZquOoLQahMrWH).

# Train
* For Flickr30k, run `train_flickr_word2_1_pool.m` for **Stage I** training.

Run `train_flickr_word_Rankloss_shift_hard` for **Stage II** training.

* For MSCOCO, run `train_coco_word2_1_pool.m` for **Stage I** training.

Run `train_coco_Rankloss_shift_hard.m` for **Stage II** training.

* For CUHK-PEDES, run `train_cuhk_word2_1_pool.m` for **Stage I** training.

Run `train_cuhk_word_Rankloss_shift` for **Stage II** training.

# Test
Select one model and have fun!

* For Flickr30k, run `test/extract_pic_feature_word2_plus_52.m` and to extract the feature from image and text. Note that you need to change the model path in the code. 

* For MSCOCO, run `test_coco/extract_pic_feature_word2_plus.m` and to extract the feature from image and text. Note that you need to change the model path in the code. 

* For CUHK-PEDES, run `test_cuhk/extract_pic_feature_word2_plus_52.m` and to extract the feature from image and text. Note that you need to change the model path in the code. 


### CheckList
- [x] Get word2vec weight

- [x] Data Preparation (Flickr30k)
- [x] Train on Flickr30k
- [x] Test on Flickr30k

- [x] Data Preparation (MSCOCO)
- [x] Train on MSCOCO
- [x] Test on MSCOCO

- [x] Data Preparation (CUHK-PEDES)
- [x] Train on CUHK-PEDES
- [x] Test on CUHK-PEDES

- [ ] Run the code on another machine 
