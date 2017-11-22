Package version: 1.6
---------------


Disclaimer
----------
You are free to use our HGLMM/LMM code for any purpose.
This package contains external software packages, as detailed in the References section below.
Using each of these packages is according to the package's terms of use.


References
----------
Our paper:
Ben Klein, Guy Lev, Gil Sadeh, Lior Wolf.
Associating Neural Word Embeddings With Deep Image Representations Using Fisher Vectors.
IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2015
http://www.cs.tau.ac.il/~wolf/papers/Klein_Associating_Neural_Word_2015_CVPR_paper.pdf

We use the The word2vec word embedding. word2vec homepage:
https://code.google.com/p/word2vec/

This package contains VLFeat version 0.9.18. It was downloaded from:
http://www.vlfeat.org/

This package contains FastICA version 2.5. It was downloaded from:
http://research.ics.aalto.fi/ica/fastica/

We use the VGG convolutional network (AKA Oxfordnet) for image feature extraction:
http://www.robots.ox.ac.uk/~vgg/research/very_deep/

We use a modified version of CCA implementation originally published by Magnus Borga. The original code:
http://www.mathworks.com/matlabcentral/fileexchange/47496-l1mccaforssvep-demo-zip/content/L1MCCAforSSVEP_Demo/cca.m
The modified version of this code is in the file cca/cca_alg.m in this package. Usage of this file is for non-commercial purposes only, as determined by Magnus Borga.


Data
----
Following are links for downloading the datasets which we used for evaluating our models.

Pascal1K:
http://vision.cs.uiuc.edu/pascal-sentences/

Flickr8K:
http://nlp.cs.illinois.edu/HockenmaierGroup/8k-pictures.html
https://illinois.edu/fb/sec/1713398
(It is recommended to use the second link rather than the first one. In the first one some image links are broken).

Flickr30K:
http://shannon.cs.illinois.edu/DenotationGraph/

COCO:
http://mscoco.org/


Overview
--------
This package contains matlab code for reproducing the HGLMM (or GMM or LMM) Fisher vector (FV) sentence representation, as described in our paper.
The word embedding that we used was word2vec (the negative-sampling version). We got best results after applying the ICA transformation on the word2vec vectors (staying in the same dimension, 300).
The FV which obtained best results was based on HGLMM with 30 clusters.
This package also contains the code which we used for:
- VGG feature extraction, for image representation.
- Canonical Correlation Analysis (CCA), for mapping the sentence vectors and image vectors to a common vector space.
This package also contains explanation and example for using our pre-trained CCA model, trained on the COCO dataset. This is the model which is referred to as GMM+HGLMM in our paper.
The following sections contain instructions for each required step.
In the following, if a file/folder is mentioned without a full path, it means this file/folder is in the fv folder.


Compiling HGLMM/LMM mex files
-------------------------------
The code of EM algorithm and of FV computation is written in C++.
This package contains code for both Linux and Windows.
Linux code is in the folders:
- HGLMM_linux
- LMM_linux
Windows code is in the folders:
- HGLMM_win
- LMM_win
These folders contain pre-compiled matlab mex files.
If you would like to compile mex files on your machine, refer to the README.txt files within those folders.


Required code changes
---------------------
- in file data_dir_base.m: change the function so it will return the path of your folder where you want to store the data files (which we will create in the sequel).


Initial step
------------
Go to the fv folder.
Run matlab.
Call:
fv_init;


Word2vec
--------
Download the word2vec word embedding from the word2vec homepage:
https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
Unzip the bin file into your data folder.
In order to convert the bin file to matlab file, call (in matlab):
word2vec_binfile_to_matlab;
This script also normalizes the vectors before saving them to file.
In order to create a sample (of 300,000 vectors, out of 3,000,000) of the word2vec vectors, call:
word2vec_sample;
You might want to use this sample in the next steps.


ICA
---
Here we apply the ICA transformation on the word2vec vectors.
Our experiments show that it is enough to calculate ICA on the sampled word2vec (which we created in the previous step). It will run faster, and will not cause degradation in results.
To do it, call:
calc_ica;
2 files will be created:
- A file containing the transformed vectors.
- A file containing the ICA transition matrix.
Remarks:
- If you want to perform ICA on the entire word2vec (without sampling), change the variable is_sampled to false (in the script calc_ica.m).
- If you want PCA instead of ICA, use the script calc_pca.


Computing HGLMM/LMM/GMM model
-----------------------------
In order to compute a HGLMM/LMM/GMM model on the word2vec vectors, use the function hglmm/lmm/gmm.
For example, to compute 30-clusters HGLMM on the ICAed sampled word2vec vectors, call:
hglmm(30, true, 'ica', 300);


Encoding Sentences as Fisher Vectors
------------------------------------
See the script fv_example.m.


VGG feature extraction
----------------------
We used the VGG convolutional network (Oxfordnet) for image feature extraction.
We used their 19-layer network.
We used this network via the matcaffe interface.
For instructions for how to use the VGG network, and for downloading the network weights file (VGG_ILSVRC_19_layers.caffemodel) and the layer configuration file (VGG_ILSVRC_19_layers_deploy.prototxt), please check out these links:
http://www.robots.ox.ac.uk/~vgg/research/very_deep/
https://gist.github.com/ksimonyan/3785162f95cd2d5fee77#file-readme-md

In addition, refer to the folder vgg in this package.
In this folder there is the file VGG_ILSVRC_19_layers_deploy.feature_extarct.prototxt.
This file is same as VGG's original configuration file, except that we removed the last 4 layers (since we want feature extraction rather than classification).
Also, refer to this files:
- vgg_init.m (matcaffe one-time initialization)
- vgg_image_file_to_features.m
- vgg_feature_extract.m
The code in this files is based on VGG's demo file matcaffe_demo_vgg_mean_pix.m.


CCA
---
Relevant files are in the folder cca.
See the script cca_example.m in this folder.
This script shows how to:
1. Train a CCA model.
2. Use our pre-trained CCA model, trained on the COCO dataset. This is the model which is referred to as GMM+HGLMM in our paper.

If you are interested in using our pre-trained CCA model, you have to download the following additional files, which are not part of this package (due to their size) but are available for download from the same location as this package:
- cca_model_g30i_h30i_eta_5e-05_cntr_1_sampled.mat
- sent_vec_sampled_features_g30i_h30i.mat
the content of these files is explained in cca_example.m


Citation
--------
If you use our code in your work, please cite:
Ben Klein, Guy Lev, Gil Sadeh, Lior Wolf. Associating Neural Word Embeddings With Deep Image Representations Using Fisher Vectors. CVPR 2015.
