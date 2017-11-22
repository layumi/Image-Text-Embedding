% dim_red_type: dimension reduction type: one of the strings: '', 'pca', 'ica'
% new_dim: relevant only if dim_red_type is not ''
function gmm(numClusters, is_sampled, dim_red_type, new_dim)

    fv_init;
    
    word2vec_sqrt = false;
    
    vectors_file_name = add_data_dir_base('GoogleNews_vectors_norm.mat');
    gmm_file_name = add_data_dir_base(sprintf('GoogleNews_norm_gmm_%d.mat', numClusters));
    
    if word2vec_sqrt
        vectors_file_name = fname_concat(vectors_file_name, '_sqrt');
        gmm_file_name = fname_concat(gmm_file_name, '_sqrt');
    end
    
    if is_sampled
        vectors_file_name = fname_concat(vectors_file_name, '_sampled');
        gmm_file_name = fname_concat(gmm_file_name, '_sampled');
    end

    switch dim_red_type
        case {'pca', 'ica', 'wht'}
            post_fix = sprintf('_%s_%d', dim_red_type, new_dim);
            vectors_file_name = fname_concat(vectors_file_name, post_fix);
            gmm_file_name = fname_concat(gmm_file_name, post_fix);
    end
    

    output(1, 'reading vectors file %s\n', vectors_file_name);
    load(vectors_file_name);
    output(1, 'done\n');

    output(1, 'Calculating GMM, %d clusters\n', numClusters);

    [n_means, n_covariances, n_priors] = vl_gmm(vectors_, numClusters);
    output(1, 'done\n');


    save(gmm_file_name, 'n_means', 'n_covariances', 'n_priors');

    output(1, 'saved to file: %s\n', gmm_file_name);
end
