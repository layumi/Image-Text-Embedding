fv_init;

in_file_name  = add_data_dir_base('GoogleNews-vectors-negative300.bin');

% we didn't get better results with sqrt
sqrt_flag = false;
norm_flag = true;

[strings_,vectors_] = read_word2vec_binfile(in_file_name, sqrt_flag, norm_flag);

words_file_name = add_data_dir_base('GoogleNews_words.mat');
save(words_file_name, 'strings_');
lg(1, 'created file: %s\n', words_file_name);


vectors_file_name = add_data_dir_base('GoogleNews_vectors_norm.mat');

if sqrt_flag
	vectors_file_name = fname_concat(vectors_file_name, '_sqrt');
end

save(vectors_file_name, 'vectors_', '-v7.3');
lg(1, 'created file: %s\n', vectors_file_name);
