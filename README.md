# Dual-Path Convolutional Image-Text Embedding with Instance Loss

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
You may download the three trained models from ~~[GoogleDrive](https://drive.google.com/open?id=1QxIdJd3oQIJSVVlAxaIZquOoLQahMrWH)~~ [new GoogleDrive](https://drive.google.com/file/d/165nn3Q07nRPGaLqPCCxPI-K5Vi9yNhYu/view?usp=share_link).

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

## 🔗 Related Projects

<p align="center"><i>Explore our ecosystem for Object Re-Identification 🔍</i></p>

<div align="center">
  <table>
    <tr>
      <td align="center" colspan="3">
        <a href="https://github.com/layumi/Person_reID_baseline_pytorch">
          <h3>⛹️</h3>
          <b>Person re-ID Baseline (PyTorch)</b>
        </a>
        <br><sub>A <b>Tiny, Friendly & Strong</b> PyTorch Baseline for Person / Vehicle Re-ID<br>with Hands-on Tutorial · The Community Standard since 2017</sub>
        <br><br>
        <a href="https://github.com/layumi/Person_reID_baseline_pytorch"><img src="https://img.shields.io/github/stars/layumi/Person_reID_baseline_pytorch.svg?style=social&label=Star" alt="GitHub stars"></a>
      </td>
    </tr>
    <tr>
      <td align="center" width="33%">
        <a href="https://github.com/layumi/3D-Magic-Mirror">
          <h3>🪞</h3>
          <b>3D Magic Mirror</b>
        </a>
        <br><sub>Clothing Reconstruction from a Single Image<br>via a Causal Perspective · npj AI'26</sub>
        <br><br>
        <a href="https://github.com/layumi/3D-Magic-Mirror"><img src="https://img.shields.io/github/stars/layumi/3D-Magic-Mirror.svg?style=social&label=Star" alt="GitHub stars"></a>
      </td>
      <td align="center" width="33%">
        <a href="https://github.com/NVlabs/DG-Net">
          <h3>✨</h3>
          <b>DG-Net</b>
        </a>
        <br><sub>Joint Generation + Re-ID Learning<br>CVPR'19 Oral · NVIDIA</sub>
        <br><br>
        <a href="https://github.com/NVlabs/DG-Net"><img src="https://img.shields.io/github/stars/NVlabs/DG-Net.svg?style=social&label=Star" alt="GitHub stars"></a>
      </td>
      <td align="center" width="33%">
        <a href="https://github.com/layumi/Person-reID_GAN">
          <h3>🎨</h3>
          <b>Person re-ID GAN</b>
        </a>
        <br><sub>GAN-based Augmentation (LSRO)<br>ICCV'17</sub>
        <br><br>
        <a href="https://github.com/layumi/Person-reID_GAN"><img src="https://img.shields.io/github/stars/layumi/Person-reID_GAN.svg?style=social&label=Star" alt="GitHub stars"></a>
      </td>
    </tr>
    <tr>
      <td align="center" width="33%">
        <a href="https://github.com/layumi/Image-Text-Embedding">
          <h3>📝</h3>
          <b>Language Person Search</b>
        </a>
        <br><sub>Text-based Person Retrieval<br>Dual-Path Embedding</sub>
        <br><br>
        <a href="https://github.com/layumi/Image-Text-Embedding"><img src="https://img.shields.io/github/stars/layumi/Image-Text-Embedding.svg?style=social&label=Star" alt="GitHub stars"></a>
      </td>
      <td align="center" width="33%">
        <a href="https://github.com/Shuyu-XJTU/APTM">
          <h3>🏷️</h3>
          <b>APTM</b>
        </a>
        <br><sub>Attribute Prompt Learning &amp; Text Matching<br>MALS Benchmark (1.5M pairs) · ACM MM'23</sub>
        <br><br>
        <a href="https://github.com/Shuyu-XJTU/APTM"><img src="https://img.shields.io/github/stars/Shuyu-XJTU/APTM.svg?style=social&label=Star" alt="GitHub stars"></a>
      </td>
      <td align="center" width="33%">
        <a href="https://github.com/Shuyu-XJTU/CMP">
          <h3>🚨</h3>
          <b>CMP</b>
        </a>
        <br><sub>Text-based Person <b>Anomaly</b> Search<br>PAB Benchmark (1M pairs) · ICCV'25 <b>Highlight</b></sub>
        <br><br>
        <a href="https://github.com/Shuyu-XJTU/CMP"><img src="https://img.shields.io/github/stars/Shuyu-XJTU/CMP.svg?style=social&label=Star" alt="GitHub stars"></a>
      </td>
    </tr>
    <tr>
      <td align="center" width="33%">
        <a href="https://github.com/layumi/person-reid-3d">
          <h3>🧊</h3>
          <b>3D Person re-ID</b>
        </a>
        <br><sub>Parameter-Efficient Re-ID<br>in the 3D Space (OG-Net)</sub>
        <br><br>
        <a href="https://github.com/layumi/person-reid-3d"><img src="https://img.shields.io/github/stars/layumi/person-reid-3d.svg?style=social&label=Star" alt="GitHub stars"></a>
      </td>
      <td align="center" width="33%">
        <a href="https://github.com/layumi/Pedestrian_Alignment">
          <h3>🚶</h3>
          <b>Pedestrian Alignment</b>
        </a>
        <br><sub>Pedestrian Alignment Network (PAN)<br>for Robust Re-ID</sub>
        <br><br>
        <a href="https://github.com/layumi/Pedestrian_Alignment"><img src="https://img.shields.io/github/stars/layumi/Pedestrian_Alignment.svg?style=social&label=Star" alt="GitHub stars"></a>
      </td>
      <td align="center" width="33%">
        <a href="https://github.com/layumi/2016_person_re-ID">
          <h3>🔁</h3>
          <b>2-Stream Person re-ID</b>
        </a>
        <br><sub>Verification + Identification<br>Baseline</sub>
        <br><br>
        <a href="https://github.com/layumi/2016_person_re-ID"><img src="https://img.shields.io/github/stars/layumi/2016_person_re-ID.svg?style=social&label=Star" alt="GitHub stars"></a>
      </td>
    </tr>
  </table>
</div>

---

<p align="center">
  ⭐ If you find our projects helpful, a <b>star</b> is the best support! ⭐
</p>

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
