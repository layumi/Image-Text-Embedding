%produce fix length word input
%keep 32 word in one scentence
load('/home/zzd/nlp/word2vector_matlab/flickr30k_dictionary.mat');
fid = fopen('/home/zzd/image-txt-retrieval/Flickr30k/flickr30k-clear.txt');
tline = fgetl(fid);
w_sum = cellfun(@(x) sum(x),subset.names);

wordcnn = zeros(32,158915,'int16');
str_length = zeros(1,158915);

count=1;

m=0;
while(ischar(tline))
    disp(count);
    word_count=0;
    skip=0;
    split_tline = strsplit(tline,{'-',' ','.',',','(',')'},'CollapseDelimiters',true);
    i=1;
    while( i<=min(numel(split_tline),32+skip)  )% use sum to do hash
        word = split_tline{i};
        i = i+1;
        sub_index = find(w_sum == sum(word));
        if(isempty(sub_index))
            skip = skip+1;
            %fprintf('%s\n',word);
            continue;
        end
        ind = cellfun(@(x) strcmp(x,word),subset.names(sub_index));
        if(sum(ind(:))==0)
            skip = skip+1;
            %fprintf('%s\n',word);
            continue;
        end
        index = sub_index(ind==1);
        %tmp = w_features(:,index);
        %sentence = sentence+tmp;
        word_count = word_count+1;
        wordcnn(word_count,count) = index;
    end
    str_length(count) = numel(split_tline);
    %}
    tline = fgetl(fid);
    count = count+1;
end

fprintf('%f',mean(str_length));
save('dense_feature_word2.1.mat','wordcnn');