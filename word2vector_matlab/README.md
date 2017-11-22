## Load word2vector from Binary file

### Download Data
Please download [GoogleNews-vectors-negative300.bin.gz](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit) first.
We decompress it.
```
(bash) gunzip GoogleNews-vectors-negative300.bin.gz
```

### Convert to Mat file for loading
We use the following script in matlab.
```
(MATLAB) read_word2vec
```
Now we extract `GoogleNews_vectors.mat` (300x3,000,000) and `GoogleNews_words.mat` (1x3,000,000)
