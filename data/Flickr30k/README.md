## Prepare Data

### Download Data
Please visit http://shannon.cs.illinois.edu/DenotationGraph/ 

### Get dicitionary
```
(matlab) make_dictionary
```

### Transfer sentence to the index in dictionary
In this step, we also get rid of rare words, which are not included in GoogleNews word2vector.
```
(matlab) prepare_wordcnn_feature2
```

### Put image url/label and text data/label in one file
```
(matlab) prepare_data
```
