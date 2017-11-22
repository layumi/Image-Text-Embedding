% This script reads word2vec bin file

addpath('./hglmm_fv_v1.6/utilities');
addpath('./hglmm_fv_v1.6/fv');
f_name = 'GoogleNews-vectors-negative300.bin';

% some settings
sqrt_flag = false;
norm_flag = false;

[w_names, w_features] = read_word2vec_binfile(f_name, sqrt_flag, norm_flag);

save('GoogleNews_words.mat','w_names');
save('GoogleNews_vectors.mat','w_features','-v7.3');