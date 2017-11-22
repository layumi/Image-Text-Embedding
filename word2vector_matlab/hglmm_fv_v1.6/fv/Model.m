% this class computes sentence-to-vec model, given word-to-vec model.
classdef Model < handle
    properties(Constant)
        MODE_AVG      = 0   % encode by average
        MODE_FISHER   = 1   % encode by fisher vectors
        MODE_BOW      = 2   % bag of words
        
        % improved fv options
        
        POWER_OFF           = 0   % no power transform
        POWER_SQRT_ALL      = 1   % sqrt all coordinates
        POWER_SQRT_GMM_ONLY = 2   % sqrt only the gmm coordinates
		POWER_EXPERIMENTAL  = 3
        
        NORMALIZE_OFF             = 0 % no normalization
        NORMALIZE_L2_ALL          = 1 % L2 normalization
        NORMALIZE_L1_L2_SELECTIVE = 2 % L2 for gmm coordinates, L1 for lmm
    end
    
    properties
        
        % improved fv options
        power_option;
        norm_option;
        
        % Word2vec model
        w2v;
        
        % GMM / LMM model
        mm_type
        mm_num_of_centers
        
        % will be used for GMM / LMM
        mm_means
        mm_covariances
        mm_priors
        
        % will be used for HGLMM
        hglmm_m
        hglmm_s
        hglmm_mu
        hglmm_sigma
        hglmm_b
        hglmm_prior

        % will be used for GG
        gg_p
        gg_mu
        gg_sigma
        
        % options paramter for gg_fv function
        gg_options
        
        % multiplication factor (workaround for numeric issues with LMM / HGLMM)
        mult_fctr;
        
        % transition matrix, for PCA / ICA
        trans_mat;
        
    end

    
    methods
	
        % constructor
        % mm_type: one of: 'gmm', 'lmm', 'hcg', 'hcl', 'hglmm'
        % lowercase: a flag indicating that our dictionary is in lowercase
        % (this is the case with GloVe)        
        function this = Model(words_file_name, vectors_file_name, freq_file_name, mm_file_name, mm_type, trans_matrix, lowercase, power_option, norm_option)
            
            if ~exist('lowercase', 'var')
                lowercase = false;
            end
            
            if ~exist('power_option', 'var')
                power_option = this.POWER_SQRT_ALL;
            end
            
            if ~exist('norm_option', 'var')
                norm_option = this.NORMALIZE_L2_ALL;
            end
            
            this.gg_options = [];
            
            this.power_option = power_option;
            this.norm_option = norm_option;
            
            this.trans_mat = trans_matrix;
            
			% load word2vec
            this.w2v = Str2vec(words_file_name, vectors_file_name, freq_file_name, lowercase);
            
            if ~isempty(mm_file_name)
                
                this.mm_type = mm_type;

                lg(1, 'MM type: %s\n', this.mm_type);

                % read GMM/LMM file
                lg(1, 'reading %s file %s\n', this.mm_type, mm_file_name);
                load(mm_file_name);
                lg(1, 'done\n');
                
                % if mult_factor was saved in the mm file (it is relevnt
                % for LMM / HGLMM)
                if exist('mult_factor', 'var')
                    this.mult_fctr = mult_factor;
                else
                    switch this.mm_type
                        case {'lmm', 'hglmm'}
                            % default is 10 (backward compatibility)
                            this.mult_fctr = 10;
                        otherwise
                            this.mult_fctr = 1;
                    end
                end

                switch this.mm_type
                    case 'hglmm'
                        this.hglmm_m = m_out;
                        this.hglmm_s = s_out;
                        this.hglmm_mu = mu_out;
                        this.hglmm_sigma = sigma_out;
                        this.hglmm_b = b_out;
                        this.hglmm_prior = prior_out;

                        this.mm_num_of_centers = length(this.hglmm_prior);
                        
                    case 'gg'
                        this.gg_p = gg_p;
                        this.gg_mu = gg_mu;
                        this.gg_sigma = gg_sigma;
                        
                        this.mm_num_of_centers = size(gg_mu,2);

                    % gmm or lmm
                    otherwise
                        this.mm_means = n_means;
                        this.mm_covariances = n_covariances;
                        this.mm_priors = n_priors;

                        this.mm_num_of_centers = length(this.mm_priors);
                end
            else
                lg(1, 'no MM file was provided\n');
            end
        end
        
        function set_gg_options(this, gg_options)
            this.gg_options = gg_options;
        end
        
        function str = mode2str(this, mode)
            switch mode
                case this.MODE_AVG
                    str = 'AVG';
                case this.MODE_FISHER
                    str = 'FISHER';
                case this.MODE_BOW
                    str = 'BOW';
                otherwise
                    str = 'UNKNOWN';
            end            
        end

        function d = encoding_size(this, mode)
            dim = size(this.trans_mat, 1);
            
            % trans_mat is empty
            if dim == 0
                dim = this.w2v.dimension;
            end
            
            switch mode
                case this.MODE_AVG
                    d = dim;
                case this.MODE_FISHER
                    switch this.mm_type
                        case 'gg'
                            d = 2 * this.mm_num_of_centers * dim;
                        otherwise
                            d = 2 * this.mm_num_of_centers * dim;
                    end
                case this.MODE_BOW
                    d = this.w2v.num_strings;
                otherwise
                    d = 0;
            end            
        end
        
        
        % encodes a sentence using fisher vectors
        function encoding = encode_sen_fv(this, sentence, num_infreq_words, sum_pair_mode)
        
			if ~exist('sum_pair_mode', 'var')
				sum_pair_mode = false;
            end
            
            X = this.w2v.sen_to_word_vecs(sentence, num_infreq_words);
            if isempty(X)
                encoding = [];
                return;
            end
            
            % encode the sentence using fisher vectors

            if sum_pair_mode
                Z = zeros(size(X,1), size(X,2) - 1);
                for i = 1:size(Z,2)
                    Z(:,i) = X(:,i) + X(:,i+1);
                end

                X = [X, Z];
            end

            if ~isempty(this.trans_mat)
                X = this.trans_mat * X;

                % in pca and ica, we normalized the vectors after
                % transformation
                X = normc(X);
            end
            

            switch this.mm_type
                case {'gmm', 'hcg'}
                    
					switch this.power_option
                        case {this.POWER_OFF, this.POWER_EXPERIMENTAL}
                            error('need to add support for POWER_OFF, POWER_EXPERIMENTAL');
                    end
                    
                    if this.norm_option == this.NORMALIZE_OFF
                        error('need to add support for NORMALIZE_OFF');
                    end
                    
                    encoding = vl_fisher(X, this.mm_means, this.mm_covariances, this.mm_priors, 'SquareRoot', 'Normalized');
                    
                case 'gg'
                    
					switch this.power_option
                        case {this.POWER_OFF, this.POWER_EXPERIMENTAL}
                            error('need to add support for POWER_OFF, POWER_EXPERIMENTAL');
                    end
                    
                    if this.norm_option == this.NORMALIZE_OFF
                        error('need to add support for NORMALIZE_OFF');
                    end
                    
                    encoding = gg_fv(X, this.gg_p, this.gg_mu, this.gg_sigma, this.gg_options);

                case {'lmm', 'hcl'}
                    encoding = FVMex(double(this.mult_fctr * X'),double(this.mm_means), double(this.mm_covariances), double(this.mm_priors),1.0);

                    switch this.power_option
                        case this.POWER_SQRT_ALL
                            encoding = sign(encoding) .* sqrt(abs(encoding));
                        case this.POWER_EXPERIMENTAL
                            %encoding = sign(encoding) .* (abs(encoding).^experimental_power);
                            %encoding = sign(encoding) .* log(abs(encoding) + 1);
                            encoding = asinh(encoding);
                    end
                        
                    switch this.norm_option
                        case this.NORMALIZE_L2_ALL
                            encoding = encoding / norm(encoding, 2);
                        case this.NORMALIZE_L1_L2_SELECTIVE
                            encoding = encoding / norm(encoding, 1);
                    end

                case 'hglmm'
                    numOfCpus = 8;

                    encoding = HybridFV(double(this.mult_fctr * X'), this.hglmm_m, this.hglmm_s, this.hglmm_mu, this.hglmm_sigma, this.hglmm_b, this.hglmm_prior, numOfCpus);

                    I_lmm = this.hglmm_b';
                    I_lmm = I_lmm(:);
                    I_lmm = [I_lmm; I_lmm];
                    I_gmm = ~I_lmm;
                    
                    switch this.power_option
                        case this.POWER_OFF
                            I_power = false(length(encoding), 1);
                        case this.POWER_SQRT_ALL
                            I_power = true(length(encoding), 1);
                        case this.POWER_SQRT_GMM_ONLY
                            I_power = I_gmm;
                    end
                        
                    switch this.norm_option
                        case this.NORMALIZE_OFF
                            I_norm2 = false(length(encoding), 1);
                            I_norm1 = false(length(encoding), 1);
                        case this.NORMALIZE_L2_ALL
                            I_norm2 = true(length(encoding), 1);
                            I_norm1 = false(length(encoding), 1);
                        case this.NORMALIZE_L1_L2_SELECTIVE
                            I_norm2 = I_gmm;
                            I_norm1 = I_lmm;
                    end
                    
                    if this.power_option == this.POWER_EXPERIMENTAL
                        %encoding = sign(encoding) .* (abs(encoding).^experimental_power);
                        %encoding = sign(encoding) .* log(abs(encoding) + 1);
                        encoding = asinh(encoding);
                    else
                        encoding(I_power) = sign(encoding(I_power)) .* sqrt(abs(encoding(I_power))); 
                    end
                    
                    encoding(I_norm2) = encoding(I_norm2) / norm(encoding(I_norm2),2);
                    encoding(I_norm1) = encoding(I_norm1) / norm(encoding(I_norm1),1);
                    
                otherwise
                    error('illegal mm_type');
            end
        end

        % encodes a sentence using average
        function encoding = encode_sen_avg(this, sentence, num_infreq_words)
            
            X = this.w2v.sen_to_word_vecs(sentence, num_infreq_words);
            if isempty(X)
                encoding = [];
                return;
            end

            if ~isempty(this.trans_mat)
                X = this.trans_mat * X;

                % in pca and ica, we normalized the vectors after
                % transformation
                X = normc(X);
            end

            % encode the sentence using average
            encoding = sum(X, 2) / size(X, 2);
            % normalize
            encoding = norma(encoding);
        end
        
        % bag of words (normalized)
        function bow = encode_sen_norm_bow(this, sentence, num_infreq_words)
            if ~isempty(num_infreq_words)
                error('encode_sen_norm_bow does not support num_infreq_words yet');
            end
            
            valid_words_in_sen = this.w2v.all_valid_words(sentence, false);
            num_valid_words_in_sen = length(valid_words_in_sen);
            
            bow = zeros(this.w2v.num_strings, 1);

            for j = 1:num_valid_words_in_sen
                word = valid_words_in_sen{j};
                word_idx = this.w2v.str2idx(word);
                bow(word_idx) = bow(word_idx) + 1;
            end
            
            % normalize
            bow = norma(bow);
        end

        % 'sentence' may be either a string, or a vector with words indices
        % (according to their order in the word2vec) of the valid words of
        % a sentence. for MODE_BOW, the words indices option is currenlty
        % not supported.
        function encoding = encode_sen(this, mode, sentence, num_infreq_words)
            
            if ~exist('num_infreq_words', 'var')
                num_infreq_words = [];
            end
                    
            switch mode
                case this.MODE_FISHER
                    encoding = this.encode_sen_fv(sentence, num_infreq_words);
                case this.MODE_AVG
                    encoding = this.encode_sen_avg(sentence, num_infreq_words);
                case this.MODE_BOW
                    if ~ischar(sentence)
                        error('for MODE_BOW, the words indices option is currenlty not supported');
                    end
                    encoding = this.encode_sen_norm_bow(sentence, num_infreq_words);
                otherwise
                    error('illegal mode');
            end
        end
        
        % words_I: a vector of indices of valid words (return by
        % all_valid_words with paramter word_indices==true).
        % this function encodes each word as an entire sentence and retures
        % the matrix of all encodings
        function words_encoding = encode_words_as_sentences(this, mode, words_I)
            num_words = length(words_I);
            
            words_encoding = zeros(this.encoding_size(mode), num_words);
            
            for i = 1:num_words
                words_encoding(:,i) = this.encode_sen(mode, words_I(i));
            end
        end
        
        
        % returns a matrix containing the encodings of the given sentences.
        % I_bad: indicators for sentences which could not be encoded since
        % none of their words is in the vocabulary. in such cases, the
        % vector of the sentence will be all zeros
        function [encoding, I_bad] = encode_sentences(this, mode, sentences)
            % if only one sentence - make it a list
            if ~iscell(sentences)
                sentences = {sentences};
            end
            
            num_sentences = length(sentences);

            encoding = zeros(this.encoding_size(mode), num_sentences);
            I_bad = false(1, num_sentences);

            lg(1, 'encoding %d sentences...\n', num_sentences);

            for i = 1:num_sentences
                v = this.encode_sen(mode, sentences{i});
                
                if isempty(v)
                    I_bad(i) = true;
                    lg(0, '\n** Sentence number %d could not be encoded since none of its words is in the vocabulary:\n', i);
                    lg(0, '%s\n', sentences{i});
                else
                    encoding(:,i) = v;
                end
                
                printmark(i, 100, num_sentences)
            end

            lg(1, 'done\n');
        end
        
    end
    
    methods(Static)
        
        function str = power2str(power_option)
            switch power_option
                case Model.POWER_OFF
                    str = 'POWER_OFF';
                case Model.POWER_SQRT_ALL
                    str = 'POWER_SQRT_ALL';
                case Model.POWER_SQRT_GMM_ONLY
                    str = 'POWER_SQRT_GMM_ONLY';
                case Model.POWER_EXPERIMENTAL
                    str = 'POWER_EXPERIMENTAL';
                otherwise
                    str = 'UNKNOWN';
            end            
        end
        
        function str = normalize2str(normalize_option)
            switch normalize_option
                case Model.NORMALIZE_OFF
                    str = 'NORMALIZE_OFF';
                case Model.NORMALIZE_L2_ALL
                    str = 'NORMALIZE_L2_ALL';
                case Model.NORMALIZE_L1_L2_SELECTIVE
                    str = 'NORMALIZE_L1_L2_SELECTIVE';
                otherwise
                    str = 'UNKNOWN';
            end            
        end
        
    end
    
end
