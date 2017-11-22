% dim_red_type: dimension reduction type: one of the strings: '', 'pca', 'ica'
% new_dim: relevant only if dim_red_type is not ''
function lmm(numClusters, is_sampled, dim_red_type, new_dim, numOfEMIter, numOfCpus)

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
    output_file_name = add_data_dir_base(sprintf('GoogleNews_norm_lmm_%d.mat', numClusters));
    
    if word2vec_sqrt
        vectors_file_name = fname_concat(vectors_file_name, '_sqrt');
        output_file_name = fname_concat(output_file_name, '_sqrt');
    end
    
    if is_sampled
        vectors_file_name = fname_concat(vectors_file_name, '_sampled');
        output_file_name = fname_concat(output_file_name, '_sampled');
    end

    if strcmp(dim_red_type,'pca') || strcmp(dim_red_type,'ica')
        post_fix = sprintf('_%s_%d', dim_red_type, new_dim);
        vectors_file_name = fname_concat(vectors_file_name, post_fix);
        output_file_name = fname_concat(output_file_name, post_fix);
    end
    
    output(1, 'reading vectors file %s\n', vectors_file_name);
    load(vectors_file_name);
    output(1, 'done\n');

    output(1, 'Calculating LMM, %d clusters\n', numClusters);
    
    % multiply by LMM_HGLMM_MULT_FACTOR=10 due to numerical issue
    % transpose because the rows should be the samples
    vectors_ = mult_factor * vectors_';

    % create a random initialization
    initPriors = (1.0 / numClusters) * ones(numClusters,1);
    muInit=vectors_(randsample(size(vectors_,1),numClusters),:);
    init_b=repmat(sum(abs(vectors_ - repmat(median(vectors_),size(vectors_,1),1))) / size(vectors_,1),numClusters,1);
    
    % calculate LMM

    output(1, 'numOfCpus = %d\n', numOfCpus);
    output(1, 'numOfEMIter = %d\n', numOfEMIter);
    
    [n_means,n_covariances,n_priors,samplesWeightsOut]=LMMMex(double(vectors_), double(muInit), double(init_b), double(initPriors), numOfEMIter, 0.01, numOfCpus);

    save(output_file_name, 'n_means', 'n_covariances', 'n_priors', 'mult_factor', '-v7.3');

    output(1, 'saved to file: %s\n', output_file_name);
end
