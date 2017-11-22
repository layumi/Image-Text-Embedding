% sentences_list: cell array of sentences (strings) to be encoded
% type: a string containig the MM type and number of clusters. e.g:
%   'hglmm_30', 'gmm_10', 'lmm_1'
%   for baseline, you can use 'avg' (average pooling)
% dim_red_type: a string containig the type of dimension reduction
%   transformation, and the new dimension (after transformation). for 
%   example:  'ica_300', 'pca_256'. use '' for no transformation
% is_sampled: a flag indicating whether the MM file was created on a sample
%   of word2vec (as suggested in the readme file)
function sent_vecs = encode_sentences(sentences_list, type, dim_red_type, is_sampled)

    fv_init;

    if ~isempty(dim_red_type)
        trans_matrix_file_name = add_data_dir_base(sprintf('trans_matrix_%s.mat', dim_red_type));
        load(trans_matrix_file_name);
        output(1, 'loaded file: %s\n', trans_matrix_file_name);
    else
        trans_matrix = [];
    end
    
    
    switch type
        case 'avg'
            mode = Model.MODE_AVG;
            mm_file_name = '';
            mm_type = '';
        case 'bow'
            mode = Model.MODE_BOW;
            mm_file_name = '';
            mm_type = '';
        otherwise
            mode = Model.MODE_FISHER;
            mm_file_name = get_mm_file_name(type, dim_red_type, is_sampled);
            C = strsplit(type, '_');
            mm_type = C{1};
    end
    

    words_file_name = add_data_dir_base('GoogleNews_words.mat');
    % normalized vectors
    vectors_file_name = add_data_dir_base('GoogleNews_vectors_norm.mat');
    
        
    m = Model(words_file_name, vectors_file_name, '', mm_file_name, mm_type, trans_matrix);

    
    [sent_vecs, I_bad] = m.encode_sentences(mode, sentences_list);

    bad_sentences = sum(I_bad);

    if bad_sentences > 0
        error('%d sentences could not be encoded since none of their words is in the vocabulary', bad_sentences);
    end
end
