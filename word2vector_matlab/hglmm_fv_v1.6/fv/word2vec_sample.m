vectors_file_name = add_data_dir_base('GoogleNews_vectors_norm.mat');

word2vec_sqrt = false;
if word2vec_sqrt
	vectors_file_name = fname_concat(vectors_file_name, '_sqrt');
end

desired_num_vectors = 300000;

% read vectors file
output(1, 'reading vectors file %s\n', vectors_file_name);
load(vectors_file_name);

num_vectors = size(vectors_, 2);

% choose rows randomly uniformly
rand_cols_idx = randsample(num_vectors, desired_num_vectors);
vectors_ = vectors_(:, rand_cols_idx);


% save to file
new_file_name = fname_concat(vectors_file_name, '_sampled');
save(new_file_name, 'vectors_');

output(1, 'saved to file: %s\n', new_file_name);
