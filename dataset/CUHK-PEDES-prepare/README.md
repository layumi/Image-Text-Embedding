## Prepare Data

### 1. Download Data
Please visit http://xiaotong.me/static/projects/person-search-language/dataset.html  and  download images of CUHK-PEDES.
The alternative download link is at http://cuhk-pedes.shuanglee.me/ (You may need to contact with the author to acqurie the dataset.)

The 'reid_raw.json' is from https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description. 
This file can be used to split the train/val/test.

### 2. Put image url and text data in one file
```
(matlab) resize_image
(matlab) prepare_imdb  %you need to change the full path in this script
(matlab) prepare_test
```

### 3. Clear data and Get dicitionary
```
(matlab) make_dictionary
```

### 3. Transfer sentence to the index in dictionary
In this step, we also get rid of rare words, which are not included in GoogleNews word2vector.
```
(matlab) prepare_wordcnn_feature2
```

