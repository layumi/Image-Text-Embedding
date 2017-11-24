## Prepare Data

### 1. Download Data
Please visit http://cocodataset.org/#home  and  download MSCOCO.

* 2014 Train Images
* 2014 Val Images
* 2014 Train/Val annotations

### 2. Make split
We random select 5,000 images as val data, 5,000 images as test data.
```
(matlab) resize_img
(matlab) prepare_imdb
```

### 3. Get dicitionary
```
(matlab) make_dictionary
```

### 4. Transfer sentence to the index in dictionary
In this step, we also get rid of rare words, which are not included in GoogleNews word2vector.
```
(matlab) prepare_wordcnn_feature2
```
