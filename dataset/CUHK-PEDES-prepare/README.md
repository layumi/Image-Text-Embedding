## Prepare Data

### 1. Download Data
Please visit http://xiaotong.me/static/projects/person-search-language/dataset.html  and  download CUHK-PEDES.

### 2. Put image url and text data in one file
```
(matlab) prepare_imdb
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

