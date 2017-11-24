load('../../word2vector_matlab/GoogleNews_words.mat');
load('../../word2vector_matlab/GoogleNews_vectors.mat');
fid = fopen('./flickr30k-train&val.txt');
tline = fgetl(fid);
w_sum = cellfun(@(x) sum(x),w_names);
subset.is_appear = zeros(1,3000000,'single');
count=1;
while(ischar(tline))
    disp(count);
    split_tline = strsplit(tline);
    for i=1:numel(split_tline)  % use sum to do hash
        word = split_tline{i};
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
    tline = fgetl(fid);
    count = count+1;
end

sub = find(subset.is_appear>0);
subset.names = {w_names{sub}};
subset.features = w_features(:,sub);
save('flickr30k_dictionary.mat','subset');

%[idex,C] = kmeans(subset.features',1024);
%save('flickr30k_kmeans_1024.mat','idex','C');
