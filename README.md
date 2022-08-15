# Dual-Path Convolutional Image-Text Embedding

[[Paper]](https://arxiv.org/abs/1711.05535) [[Slide]](http://zdzheng.xyz/files/ZhedongZheng_CA_Talk_DualPath.pdf) :arrow_left: **I recommend to check this slide first.** :arrow_left:

This repository contains the code for our paper [Dual-Path Convolutional Image-Text Embedding](https://arxiv.org/abs/1711.05535). Thank you for your kindly attention. 

### Some News
- Instance Loss (Pytorch version) is now available at https://github.com/layumi/Person_reID_baseline_pytorch/blob/master/instance_loss.py  

**5 Sep 2021** I love the sentence that 'Define yourself via tell what you are different from others' (exemplar SVM), which also is the spirit of the instance loss. 

**11 June 2020** People live in the 3D world. We release one new person re-id code [Person Re-identification in the 3D Space](https://github.com/layumi/person-reid-3d), which conduct representation learning in the 3D space. You are welcomed to check out it.

**30 April 2020** We have won the [AICity Challenge 2020](https://www.aicitychallenge.org/) in CVPR 2020,  yielding the 1st Place Submission to the retrieval track :red_car:. Check out [here](https://github.com/layumi/AICIty-reID-2020).

**01 March 2020** We release one new image retrieval dataset, called [University-1652](https://github.com/layumi/University1652-Baseline), for drone-view target localization and drone navigation :helicopter:. It has a similar setting with the person re-ID. You are welcomed to check out it.

![](http://zdzheng.xyz/images/fulls/ConvVSE.jpg)

![](https://github.com/layumi/Image-Text-Embedding/blob/master/CUHK-show.jpg)

**What's New**: We updated the paper to the second version, adding more illustration about the mechanism of the proposed instance loss.

# Install Matconvnet
I have included my Matconvnet in this repo, so you do not need to download it again.You just need to uncomment and modify some lines in gpu_compile.m and run it in Matlab. Try it~ (The code does not support cudnn 6.0. You may just turn off the Enablecudnn or try cudnn5.1)

If you fail in compilation, you may refer to http://www.vlfeat.org/matconvnet/install/

# Prepocess Datasets
1. Extract wrod2vec weights. Follow the instruction in `./word2vector_matlab`;

2. Prepocess the dataset. Follow the instruction in `./dataset`. You can choose one dataset to run.
Three datasets need different prepocessing. I write the instruction for [Flickr30k](https://github.com/layumi/Image-Text-Embedding/tree/master/dataset/Flickr30k-prepare), [MSCOCO](https://github.com/layumi/Image-Text-Embedding/tree/master/dataset/MSCOCO-prepare) and [CUHK-PEDES](https://github.com/layumi/Image-Text-Embedding/tree/master/dataset/CUHK-PEDES-prepare).

3. Download the model pre-trained on ImageNet. And put the model into './data'.
```
(bash) wget http://www.vlfeat.org/matconvnet/models/imagenet-resnet-50-dag.mat
```
Alternatively, you may try [VGG16](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-16.mat) or [VGG19](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat). 

You may have a different split with me. (Sorry, this is my fault. I used a random split.) Just for a backup, this is the [dictionary archive](https://drive.google.com/open?id=1Yp6B5GKhgQTD9bsmvmVkvxt-SnmHHjVA) used in the paper.

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

### Citation
```bibtex
@article{zheng2017dual,
  title={Dual-Path Convolutional Image-Text Embeddings with Instance Loss},
  author={Zheng, Zhedong and Zheng, Liang and Garrett, Michael and Yang, Yi and Xu, Mingliang and Shen, Yi-Dong},
  journal={ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM)},
  doi={10.1145/3383184},
  note={\mbox{doi}:\url{10.1145/3383184}},
  volume={16},
  number={2},
  pages={1--23},
  year={2020},
  publisher={ACM New York, NY, USA}
}
```
