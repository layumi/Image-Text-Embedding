vectors_file_name = add_data_dir_base('GoogleNews_vectors_norm.mat');

word2vec_sqrt = false;
if word2vec_sqrt
	vectors_file_name = fname_concat(vectors_file_name, '_sqrt');
end

is_sampled = true;
% without normalizing we had numeric problems in the LMM
normalize  = true;

if is_sampled
    vectors_file_name = fname_concat(vectors_file_name, '_sampled');
end

output(1, 'reading vectors file %s\n', vectors_file_name);
load(vectors_file_name);
output(1, 'done\n');

TargetDim = 300;

%desiredTrnSizeForICA = 512000;

output(1, 'Use ICA to reduce to dimension %d\n', TargetDim);

[icasig, A, trans_matrix] = fastica (vectors_, 'numOfIC', TargetDim);
vectors_ = trans_matrix * vectors_;

[dim, num_vectors] = size(vectors_)

if normalize
    % normalize
    output(1, 'normalizing...\n');
    for i = 1:num_vectors
      vectors_(:,i) = norma(vectors_(:,i));
    end
end

output(1, 'saving to file...\n');

post_fix = sprintf('_ica_%d', TargetDim);
ica_vectors_file_name = fname_concat(vectors_file_name, post_fix);
trans_matrix_file_name = add_data_dir_base(strcat('trans_matrix', post_fix));

save(ica_vectors_file_name, 'vectors_', '-v7.3');
output(1, 'saved to file: %s\n', ica_vectors_file_name);

save(trans_matrix_file_name, 'trans_matrix', '-v7.3');
output(1, 'saved to file: %s\n', trans_matrix_file_name);

clear vectors_;
clear trans_matrix;
