function output = rand_diff_class2(imdb,label)
%for cuhk-pede  txt diff
    index = find(imdb.images.label2~=label);
    output = index(randi(numel(index)));
    while(imdb.images.label2(output)~=0)  %if val/test, sample again
        output = index(randi(numel(index)));
    end
end