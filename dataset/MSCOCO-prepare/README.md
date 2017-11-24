## Prepare Data

### 1. Download Data
Please visit http://shannon.cs.illinois.edu/DenotationGraph/   and  download Flickr30k.

### 2. Clear data and Get dicitionary
```
(matlab) clear_txt
(matlab) make_dictionary
```

### 3. Transfer sentence to the index in dictionary
In this step, we also get rid of rare words, which are not included in GoogleNews word2vector.
```
(matlab) prepare_wordcnn_feature2
```

### 4. Put image url and text data in one file
```
(matlab) prepare_data
```
