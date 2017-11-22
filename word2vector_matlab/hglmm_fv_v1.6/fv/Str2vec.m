% this class holds a list of strings and a list of their vector
% representations, and gives simple services, such as retrieving
% the vector representation of a string.
classdef Str2vec < handle
    
    properties(Constant)
        % seperator characters for extracting words from sentences
        % the characters `‘’ are handled by function process_text
        % the characters -' are handled by method all_valid_words
        sepchars = [' .,;:?!"&()[]{}<>=/\“”',char([9 10 13])];
    end
    
    properties
        strings
        vectors
        num_strings
        dimension
        map
        respell_typo_map
        freq
        % this flag indicates that our dictionary is in lowercase (this is
        % the case with GloVe)
        lowercase
    end

    
    methods
	
        % constructor
        function this = Str2vec(strings_file_name, vectors_file_name, freq_file_name, lowercase)
            
            if exist('lowercase', 'var')
                this.lowercase = lowercase;
            else
                this.lowercase = false;
            end

            
            if isempty(vectors_file_name)
                return;
            end
            
            % read vectors file
            output(1, 'reading vectors file %s\n', vectors_file_name);
            load(vectors_file_name);
            
            % needed for SI stacked FV, where the variable name is sent_vecs
            if exist('sent_vecs', 'var')
                this.vectors = sent_vecs;
                clear sent_vecs;
            else
                this.vectors = vectors_;
                clear vectors_;
            end
            
            output(1, 'done\n');

            if exist('freq_file_name', 'var') && ~isempty(freq_file_name)
                output(1, 'reading word frequencies file %s\n', freq_file_name);
                load(freq_file_name);
                this.freq = freq;
                clear freq;
                output(1, 'done\n');
            else
                output(1, 'word frequencies file was not provided\n');
            end

            [this.dimension, this.num_strings] = size(this.vectors);

            output(0, 'vectors dimension: %d\n', this.dimension);
            output(0, 'number of strings: %d\n', this.num_strings);

            no = norm(this.vectors(:, 1));
            if abs(no - 1) > 0.1
                output(0, 'NOTICE: vectors are not normalized\n');
            end
            
            
            if isempty(strings_file_name)
                output(1, 'strings file was not provided\n');
            else
                
                % read strings file
                output(1, 'reading strings file %s\n', strings_file_name);
                load(strings_file_name);
                this.strings = strings_;
                clear strings_;
                output(1, 'done\n');

                
                % prepare map for faster lookup
                this.map = containers.Map(this.strings, 1:this.num_strings);
                

                pathstr = fileparts( mfilename('fullpath') );
                respell_typo_file_name = fullfile(pathstr, 'respell_typo.txt');
                
                if file_exist(respell_typo_file_name)
                    output(1, 'reading respell typo file %s\n', respell_typo_file_name);
                    respell_typo_text = fileread(respell_typo_file_name);
                    C = textscan(respell_typo_text, '%s %s');
                    % prepare mapping from typo to correct word
                    this.respell_typo_map = containers.Map(C{1,1}, C{1,2});
                else
                    % create "empty" map
                    this.respell_typo_map = containers.Map(' ', ' ');
                    
                    output(1, 'respell typo file does not exist: %s\n', respell_typo_file_name);
                end
                
            end            
        end
        
        function normalize(this)
            for i = 1:this.num_strings
              this.vectors(:,i) = this.vectors(:,i)./norm(this.vectors(:,i));
            end
        end
	
        function tf = exist(this, str)
            
            if this.lowercase
                str = lower(str);
            end
            
			tf = this.map.isKey(str);
        end
        
        function idx = str2idx(this, str)
            
            if this.lowercase
                str = lower(str);
            end
            
			idx = this.map(str);
        end
        
        function vec = get_vec(this, str)
            if this.exist(str)
                vec = this.vectors(:, this.str2idx(str));
            else
                vec = [];
            end
        end

        function freq = get_freq(this, str)
            if this.exist(str)
                freq = this.freq(this.str2idx(str));
            else
                freq = [];
            end
        end

        function show_strings(this, start_i, end_i)
			for i = start_i:end_i
				fprintf('%7d: %s\n', i, this.strings{i});
			end
        end

	    % cosine similarity
        function cos_sim = sim(this, str1, str2)
			v1 = this.get_vec(str1);
			v2 = this.get_vec(str2);
			%cos_sim = (v1'*v2)/(norm(v1)*norm(v2));
            cos_sim = v1'*v2; % assuming the vectors are normalized
        end
        
        function [sorted_sim, I] = sim_order_vec(this, vec)
            similarities = vec' * this.vectors;
            [sorted_sim, I] = sort(similarities, 'descend');
        end

        function [sorted_sim, I] = sim_order(this, str)
			vec = this.get_vec(str);
            [sorted_sim, I] = this.sim_order_vec(vec);
        end

        function [sorted_sim, I] = sim_order_i(this, i)
            [sorted_sim, I] = this.sim_order_vec(this.vectors(:,i));
        end
        
        % print the k nearest neighbors
        function knn_vec(this, vec, k)
            [sorted_sim, I] = sim_order_vec(this, vec);
            
            for i = 1:k
                lg(0, '%20s %.2f\n', this.strings{I(i)}, sorted_sim(i));
            end
        end

        % print the k nearest neighbors
        function knn(this, str, k)
            vec = this.get_vec(str);
            knn_vec(this, vec, k);
        end
        
        function save_to_file(this, name)
            strings_file_name = sprintf('%s_strings.mat', name);
            
            strings_ = this.strings;
            save(strings_file_name, 'strings_', '-v7.3');
            clear strings_;
            
            vectors_file_name = sprintf('%s_vectors.mat', name);
            
            vectors_ = this.vectors;
            save(vectors_file_name, 'vectors_', '-v7.3');
            clear vectors_;
        end
        
        
        % if word_indices is true, the returned value is the words indices
        % instead of list of the words
        function ret_val = all_valid_words(this, text, enable_log, filter_stopwords, word_indices)
            
            if ~exist('filter_stopwords', 'var')
                filter_stopwords = false;
            end
            
            if ~exist('word_indices', 'var')
                word_indices = false;
            end
            
            if filter_stopwords
                stopwords_map = stopwords_load();
            end
            
            text = this.process_text(text);
            
            % the try & catch are needed for the case of text
            % containing only the sepchars
            try
                initial_words_list = allwords(text, this.sepchars);
            catch
                if word_indices
                    ret_val = [];
                else
                    ret_val = {};
                end
                
                return;
            end                
            
            initial_words_list_len = length(initial_words_list);
            
            % this preallocation if for efficiency. finally, the length
            % of words_list might be larger or smaller
            words_list = cell(1, initial_words_list_len);
            words_list_idx = 1;
            
            if enable_log
                not_found_list = cell(1, initial_words_list_len);
                not_found_list_idx = 1;
            end
            
            respell_count = 0;
            subword_count = 0;
            
            for initial_words_list_idx = 1:initial_words_list_len
                
                word = initial_words_list{initial_words_list_idx};
                
                if this.exist(word)

                    if filter_stopwords && stopwords_map.isKey(word);
                        %lg(0, 'stopword %s was filtered\n', word);
                        continue;
                    end
                    
                    words_list{words_list_idx} = word;
                    words_list_idx = words_list_idx + 1;
                else
                    respelled_word = this.try_respell(word);
                    
                    if this.exist(respelled_word)
                        words_list{words_list_idx} = respelled_word;
                        words_list_idx = words_list_idx + 1;

                        if enable_log
                            lg(0, '[respell] %s -> %s\n', word, respelled_word);
                        end
                        
                        respell_count = respell_count + 1;
                        
                        continue;
                    end
                    
                    
                    % try to seperate w.r.t. sepchars - and '
                    % the try & catch are needed for the case of words
                    % containing only these sepchars
                    try
                        sub_words = allwords(word, '-+''0123456789');
						%sub_words = allwords(word, '-''');
                    catch
                        continue;
                    end
                    
                    not_found_at_all = true;
                    
                    for i = 1:length(sub_words)
                        sub_word = sub_words{i};
						
						sub_word_exist = this.exist(sub_word);
						if ~sub_word_exist
							sub_word = this.try_respell_typo(sub_word);
							sub_word_exist = this.exist(sub_word);
						end
                        
                        if sub_word_exist
                            
                            switch sub_word
                                % sub-words that we don't want
                                case {'s'}
                            
                                otherwise
                                    
                                    not_found_at_all = false;
                                    
                                    words_list{words_list_idx} = sub_word;
                                    words_list_idx = words_list_idx + 1;

                                    if enable_log
                                        lg(0, '[subword] %s (full word was: %s)\n', sub_word, word);
                                    end
                            end
                        end
                    end
                    
                    if not_found_at_all
                        if enable_log
                            switch word
                                % these words will cause a lot of printings
                                case {'a', 'and', 'of', 'to', '''s'}

                                otherwise
                                    lg(0, '[-------] %s (not found)\n', word);
                                    
                                    not_found_list{not_found_list_idx} = word;
                                    not_found_list_idx = not_found_list_idx + 1;
                            end
                        end
                    else
                        subword_count = subword_count + 1;
                    end
                    
                end
            end
            
            len = words_list_idx-1;
            
            words_list = words_list(1:len);
            
            if word_indices
                ret_val = zeros(len, 1);
                
                for i = 1:len
                    ret_val(i) = this.str2idx(words_list{i});
                end
            else
                ret_val = words_list;
            end
            
            
            if enable_log
                not_found_list = unique( not_found_list(1 : not_found_list_idx-1) );
                num_not_found = length(not_found_list);
                
                lg(1, 'not found unique words:\n');
                for i = 1:num_not_found
                    lg(0, '%s\n', not_found_list{i});
                end
                print_newline();
                
                lg(1, 'respell_count:     %d\n', respell_count);
                lg(1, 'subword_count:     %d\n', subword_count);
                lg(1, 'total saved words: %d\n', respell_count + subword_count);
                lg(1, 'not found unique words: %d\n', num_not_found);
            end
        end
        
        
        % 'sentence' may be either a string, or a vector with words indices
        % (according to their order in the word2vec) of the valid words of
        % a sentence.
        % if num_infreq_words is passed and is in not empty, only (up to)
        % num_infreq_words words of the sentence will be used, and the
        % other words will be ignored.
        function X = sen_to_word_vecs(this, sentence, num_infreq_words)
            
            if ischar(sentence)
                words_in_sen = this.all_valid_words(sentence, false);
            else
                words_in_sen = sentence;
            end
            
            num_words_in_sen = length(words_in_sen);
            
            if num_words_in_sen == 0
                X = [];
                return;
            end
            
            if exist('num_infreq_words', 'var') && ~isempty(num_infreq_words) && num_infreq_words ~= 0
                if ~ischar(sentence)
                    error('num_infreq_words currently not supported when paramter "sentence" contains word indices');
                end
                
                freqs = zeros(1, num_words_in_sen);

                for j = 1:num_words_in_sen
                    % no need to verify that freq is not empty, we already know
                    % that all words in words_in_sen are valid.
                    freqs(j) = this.get_freq(words_in_sen{j});
                end
                
                [~, I] = sort(freqs);
                
                % take the most non-frequent words
                num_words_in_sen = min(num_words_in_sen, num_infreq_words);
                words_in_sen = words_in_sen(I(1:num_words_in_sen));
            end
            

            % preallocate matrix for the vectors of the words of the
            % sentence
            X = zeros(this.dimension, num_words_in_sen);

            % fill the matrix X
            for j = 1:num_words_in_sen
                if ischar(sentence)
                    % no need to verify that vec is not empty, we already know
                    % that all words in words_in_sen are valid.
                    X(:, j) = this.get_vec(words_in_sen{j});
                else
                    X(:, j) = this.vectors(:, words_in_sen(j));
                end
            end
        end
        
        
        function respelled_word = try_respell_dash(this, word)
            if length(word) >= 3
                k = strfind(word, '-');
                if length(k) == 1
                    respelled_word = [word(1:k-1) word(k+1:end)];
                    
                    if ~this.exist(respelled_word)
                        respelled_word = strrep(word, '-', '_');
                    end
                    return;
                end
            end

            respelled_word = word;
        end


        function respelled_word = try_respell_apostrophe(this, word)
            if length(word) >= 3 && strcmp(word(end-1:end), '''s')
                respelled_word = word(1:end-2);
            else
                respelled_word = word;
            end
        end


        function respelled_word = try_respell_typo(this, word)
            if this.respell_typo_map.isKey(word);
                respelled_word = this.respell_typo_map(word);
            else
                respelled_word = word;
            end
        end
        
        
        function respelled_word = try_respell_uk2us(this, word)
            respelled_word = strrep(word, 'our', 'or');
            respelled_word = strrep(respelled_word, 'ise', 'ize');
            respelled_word = strrep(respelled_word, 'ising', 'izing');
            respelled_word = strrep(respelled_word, 'isation', 'ization');
        end


        function respelled_word = try_respell(this, word)
            word = this.try_respell_apostrophe(word);
            if ~this.exist(word)
                word = this.try_respell_dash(word);
                if ~this.exist(word)
                    word = this.try_respell_typo(word);
                    if ~this.exist(word)
                        word = this.try_respell_uk2us(word);
                    end
                end
            end
            
            respelled_word = word;
        end
        
        function processed_str = process_text(this, str)
            
            str = strrep(str, '“', '"');
            str = strrep(str, '”', '"');
            str = strrep(str, '—', '-');
            str = strrep(str, '´', '''');
            str = strrep(str, '`', '''');
            str = strrep(str, '‘', '''');
            str = strrep(str, '’', '''');
            str = strrep(str, 'Â', '''');
            str = strrep(str, 'â€?', ' ');
            str = strrep(str, 'â€?', ' ');
            str = strrep(str, 'â€˜â€˜', '"');
            str = strrep(str, 'â€™', '''');
            str = strrep(str, 'â€"', '-');
            
            % phrases / missing space
            str = strrep(str, 'Force Ouvriere', 'Force_Ouvriere');
            str = strrep(str, '$US', '$');
            str = strrep(str, 'US$', '$');
            str = strrep(str, 'livingdining', 'living dining');
            str = strrep(str, 'bluegreen', 'blue green');
            str = strrep(str, 'kitchendiner', 'kitchen diner');
            str = strrep(str, 'onebrown', 'one brown');
            str = strrep(str, 'atdinner', 'at dinner');
            str = strrep(str, 'kitchendining', 'kitchen dining');
            str = strrep(str, 'jackethalter', 'jacket halter');
            str = strrep(str, 'ridinghorses', 'riding horses');
            str = strrep(str, 'backwheel', 'back wheel');
            str = strrep(str, 'withjackets', 'with jackets');
            str = strrep(str, 'playingtwister', 'playing twister');
            str = strrep(str, 'dirtbiking', 'dirt biking');
            str = strrep(str, 'windowbox', 'window box');
			str = strrep(str, 'Middle-aged', 'middle_aged');
			str = strrep(str, 'Airgame', 'Air game');
			str = strrep(str, 'Beatermix', 'Beater mix');
			str = strrep(str, 'ConstructionBike', 'Construction Bike');
			str = strrep(str, 'Craftslady', 'Crafts lady');
			str = strrep(str, 'Dirtbikers', 'Dirt bikers');
			str = strrep(str, 'dirtbikers', 'dirt bikers');
			str = strrep(str, 'dirtracing', 'dirt racing');
			str = strrep(str, 'Radio Flyre', 'Radio_Flyer');
			str = strrep(str, 'Radio Flyer', 'Radio_Flyer');
			str = strrep(str, 'Goucho Marx', 'Groucho_Marx');
			str = strrep(str, 'Groucho Marx', 'Groucho_Marx');
			str = strrep(str, 'De Koninck', 'De_Koninck');
			str = strrep(str, 'Laperla', 'La_Perla');
			str = strrep(str, 'La Perla', 'La_Perla');
			str = strrep(str, 'Kuala Lumpar', 'Kuala_Lumpar');
			str = strrep(str, 'MonsterBowl', 'Monster Bowl');
			str = strrep(str, 'MoonBounce', 'Moon_Bounce');
			str = strrep(str, 'Racedog', 'Race dog');
			str = strrep(str, 'Scantily clad', 'Scantily_clad');
			str = strrep(str, 'scantily clad', 'scantily_clad');
			str = strrep(str, 'Seattle Seahawkss', 'Seattle_Seahawks');
			str = strrep(str, 'Seattle Seahawks', 'Seattle_Seahawks');
			str = strrep(str, 'Skimply clad', 'skimpily_clad');
			str = strrep(str, 'skimpily clad', 'skimpily_clad');
			str = strrep(str, 'Skimpily clad', 'skimpily_clad');
			str = strrep(str, 'TheFaceShop', 'The Face Shop');
			str = strrep(str, 'Arc De TRiomphe', 'Arc_de_Triomphe');
			str = strrep(str, 'Arc de Triomphe', 'Arc_de_Triomphe');
			str = strrep(str, 'backlegs', 'back legs');
			str = strrep(str, 'backstand', 'back stand');
			str = strrep(str, 'backview', 'back view');
			str = strrep(str, 'basett hound', 'basset_hound');
			str = strrep(str, 'basset hound', 'basset_hound');
			str = strrep(str, 'justin beiber', 'justin_bieber');
			str = strrep(str, 'justin bieber', 'justin_bieber');
			str = strrep(str, 'Asus betbook', 'Asus_netbook');
			str = strrep(str, 'Asus netbook', 'Asus_netbook');
			str = strrep(str, 'bigwheels', 'big wheels');
			str = strrep(str, 'bluejacket', 'blue jacket');
			str = strrep(str, 'boogieboard', 'boogie board');
			str = strrep(str, 'snow borader', 'snowboarder');
			str = strrep(str, 'burbur carpet', 'berber_carpet');
			str = strrep(str, 'berber carpet', 'berber_carpet');
			str = strrep(str, 'busstop', 'bus stop');
			str = strrep(str, 'cappedhills', 'capped hills');
			str = strrep(str, 'cruisship', 'cruise_ship');
			str = strrep(str, 'cruise ship', 'cruise_ship');
			str = strrep(str, 'break danging', 'breakdancing');
			str = strrep(str, 'break dancing', 'breakdancing');
			str = strrep(str, 'john deere', 'John_Deere');
			str = strrep(str, 'John Deere', 'John_Deere');
			str = strrep(str, 'dimlight', 'dim light');
			str = strrep(str, 'diveboard', 'dive board');
			str = strrep(str, 'downsteps', 'down steps');
			str = strrep(str, 'dresswear', 'dress wear');
			str = strrep(str, 'facepaintings', 'face paintings');
			str = strrep(str, 'fingerhold', 'finger hold');
			str = strrep(str, 'foggyday', 'foggy day');
			str = strrep(str, 'handstanding', 'hand standing');
			str = strrep(str, 'tommy hilfiger', 'Tommy_Hilfiger');
			str = strrep(str, 'Tommy Hilfiger', 'Tommy_Hilfiger');
			str = strrep(str, 'Tim hortons', 'Tim_Hortons');
			str = strrep(str, 'Tim Hortons', 'Tim_Hortons');
			str = strrep(str, 'iceskate', 'ice skate');
			str = strrep(str, 'jumpropes', 'jump ropes');
			str = strrep(str, 'longeared', 'long eared');
			str = strrep(str, 'meetinghall', 'meeting hall');
			str = strrep(str, 'milkbone', 'milk bone');
			str = strrep(str, 'monoboard', 'mono board');
			str = strrep(str, 'monocycle', 'mono cycle');
			str = strrep(str, 'moter bike', 'motorbike');
			str = strrep(str, 'mudfight', 'mud fight');
			str = strrep(str, 'orangesunset', 'orange sunset');
			str = strrep(str, 'pabst blue ribbon', 'Pabst_Blue_Ribbon');
			str = strrep(str, 'Pabst Blue Ribbon', 'Pabst_Blue_Ribbon');
			str = strrep(str, 'parkinglot', 'parking lot');
			str = strrep(str, 'playgym', 'play gym');
			str = strrep(str, 'playtoy', 'play toy');
			str = strrep(str, 'pokeball', 'poke ball');
			str = strrep(str, 'policeperson', 'police person');
			str = strrep(str, 'golden retreiver', 'golden_retriever');
			str = strrep(str, 'Golden Retreiver', 'golden_retriever');
			str = strrep(str, 'golden retriever', 'golden_retriever');
			str = strrep(str, 'Golden Retriever', 'golden_retriever');
			str = strrep(str, 'riverrafting', 'river rafting');
			str = strrep(str, 'riverwater', 'river water');
			str = strrep(str, 'light saber', 'light_saber');
			str = strrep(str, 'light Sabre', 'light_saber');
			str = strrep(str, 'Sheppard dogs', 'Shepherd dogs');
			str = strrep(str, 'shoulderbag', 'shoulder bag');
			str = strrep(str, 'skiboarding', 'ski boarding');
			str = strrep(str, 'skimboarder', 'ski boarder');
			str = strrep(str, 'snowpile', 'snow pile');
			str = strrep(str, 'snowshovel', 'snow shovel');
			str = strrep(str, 'stonesign', 'stone sign');
			str = strrep(str, 'streetpole', 'street pole');
			str = strrep(str, 'streetway', 'street way');
			str = strrep(str, 'surfboarder', 'surf boarder');
			str = strrep(str, 'swimcap', 'swim cap');
			str = strrep(str, 'throughwindow', 'through window');
			str = strrep(str, 'treefilled', 'tree filled');
			str = strrep(str, 'uptop', 'up top');
			str = strrep(str, 'G ?uys', 'Guys');
			str = strrep(str, 'pole vaulated', 'pole vaulted');
			str = strrep(str, 'vike helmets', 'bike helmets');
			str = strrep(str, 'waveboarder', 'wave boarder');
			str = strrep(str, 'weightlifted', 'weight lifted');
			str = strrep(str, 'windboard', 'wind board');
			str = strrep(str, 'windboarder', 'wind boarder');
			str = strrep(str, 'windsailing', 'wind sailing');
			str = strrep(str, 'ca n''t', 'can''t');
			str = strrep(str, 'darked skinned', 'dark_skinned');
			str = strrep(str, 'dark skinned', 'dark_skinned');
			
            processed_str = str;
        end
        
    end
    
end
