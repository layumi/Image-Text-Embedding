load('../../word2vector_matlab/GoogleNews_words.mat');
load('../../word2vector_matlab/GoogleNews_vectors.mat');
load('caption_train.mat');
w_sum = cellfun(@(x) sum(x),w_names);
subset.is_appear = zeros(1,3000000,'single');

for i = 1:numel(caption_dic)
    disp(i);
    split_tline = strsplit(caption_dic{i},{'-',' ','.',',','(',')'},'CollapseDelimiters',true);
    for j=1:numel(split_tline)  % use sum to do hash
        word = split_tline{j};
        sub_index = find(w_sum == sum(word));
        if(isempty(sub_index))
            continue;
        end
        ind = cellfun(@(x) strcmp(x,word),w_names(sub_index));
        if(sum(ind(:))==0)
            continue;
        end
        index = sub_index(ind==1);
        %tmp = w_features(:,index);
        %sentence = sentence+tmp;
        subset.is_appear(index) = subset.is_appear(index) +1;
    end
end

sub = find(subset.is_appear>0);
subset.names = {w_names{sub}};
subset.features = w_features(:,sub);
save('CUHK-PEDES_dictionary.mat','subset');
