% example sentences list
sent1 = 'first example sentence';
sent2 = 'second example sentence';
sent3 = 'third example sentence';
sentences_list = {sent1, sent2, sent3};


% here we encode the sentences with FV based on HGLMM with 30 clusters.
% you can replace hglmm by gmm or lmm, and you can replace the 30 by any
% integer number (provided that you had created a corresponding MM file)
type = 'hglmm_30';

% in this example we use HGLMM computed on 300-dimensional word2vec after
% the ICA transformation.
% examples for possilbe values: 'ica_300', 'pca_300', '' (the latter means
% no transformation).
dim_red_type = 'ica_300';

% in this example we use HGLMM model calculated on sampled word2vec
is_sampled = true;

% encode the sentences
hglmm_30_ica_sent_vecs = encode_sentences(sentences_list, type, dim_red_type, is_sampled);


% now encode the sentences with FV based on GMM with 30 clusters.
% (still, with word2vec after the ICA transformation)
type = 'gmm_30';
dim_red_type = 'ica_300';
is_sampled = true;
gmm_30_ica_sent_vecs = encode_sentences(sentences_list, type, dim_red_type, is_sampled);


% now encode the sentences using the average pooling baseline
type = 'avg';
dim_red_type = 'ica_300';
is_sampled = true;
sent_vecs_baseline = encode_sentences(sentences_list, type, dim_red_type, is_sampled);
