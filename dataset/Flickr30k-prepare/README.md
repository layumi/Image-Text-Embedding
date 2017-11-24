## Prepare Data

### 1. Download Data
Please visit http://shannon.cs.illinois.edu/DenotationGraph/   and  download Flickr30k.

### 2. Split Data set (Train/Val/Test)
```
(matlab) split_Flickr30k
(matlab) resize_image
(matlab) prepare_imdb
```

### 3. Clear training data and Get dicitionary
```
(matlab) train_txt
(matlab) make_dictionary
```

### 3. Transfer sentence to the index in dictionary
In this step, we also get rid of rare words, which are not included in GoogleNews word2vector.
```
(matlab) clear_txt
(matlab) prepare_wordcnn_feature2
```
