% dim_red_type: dimension reduction type: one of the strings: '', 'pca', 'ica'
% new_dim: relevant only if dim_red_type is not ''
function hglmm(numClusters, is_sampled, dim_red_type, new_dim, numOfEMIter, numOfCpus)

    fv_init;

    if ~exist('numOfEMIter', 'var')
        numOfEMIter = 10;
    end
    
    if ~exist('numOfCpus', 'var')
        numOfCpus = 4;
    end
    
    word2vec_sqrt = false;
    
    mult_factor = LMM_HGLMM_MULT_FACTOR;
    
    output(1, 'LMM_HGLMM_MULT_FACTOR = %d\n', LMM_HGLMM_MULT_FACTOR);

    vectors_file_name = add_data_dir_base('GoogleNews_vectors_norm.mat');
    output_file_name = add_data_dir_base(sprintf('GoogleNews_norm_hglmm_%d.mat', numClusters));
    
    if word2vec_sqrt
        vectors_file_name = fname_concat(vectors_file_name, '_sqrt');
        output_file_name = fname_concat(output_file_name, '_sqrt');
    end
    
    if is_sampled
        vectors_file_name = fname_concat(vectors_file_name, '_sampled');
        output_file_name = fname_concat(output_file_name, '_sampled');
    end

    switch dim_red_type
        case {'pca', 'ica', 'wht'}
            post_fix = sprintf('_%s_%d', dim_red_type, new_dim);
            vectors_file_name = fname_concat(vectors_file_name, post_fix);
            output_file_name = fname_concat(output_file_name, post_fix);
    end
    
    output(1, 'reading vectors file %s\n', vectors_file_name);
    load(vectors_file_name);
    output(1, 'done\n');

    output(1, 'Calculating HGLMM, %d clusters\n', numClusters);
    
    % multiply by LMM_HGLMM_MULT_FACTOR=10 due to numerical issue
    % transpose because the rows should be the samples
    vectors_ = mult_factor * vectors_';

    % calculate HGLMM
    
    output(1, 'numOfCpus = %d\n', numOfCpus);
    output(1, 'numOfEMIter = %d\n', numOfEMIter);
    
    [m_out, s_out, mu_out, sigma_out, b_out, prior_out, samplesWeightsOut ]=HybridEM(vectors_, numClusters, numOfEMIter, numOfCpus);

    save(output_file_name, 'm_out', 's_out', 'mu_out', 'sigma_out', 'b_out', 'prior_out', 'mult_factor', '-v7.3');

    output(1, 'saved to file: %s\n', output_file_name);
    
    num_lmm = sum(sum(b_out));
    [rows, cols] = size(b_out);
    lmm_fraction = num_lmm / ( rows*cols );
    output(1, 'num_lmm: %d\n', num_lmm);
    output(1, 'lmm_fraction: %.3f\n', lmm_fraction);
    
end
