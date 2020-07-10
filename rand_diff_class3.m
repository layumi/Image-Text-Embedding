function output = rand_diff_class3(imdb,label)
%for cuhk-pede  img diff
    index = find(imdb.images.label~=label);
    output = index(randi(numel(index)));
    %while(imdb.images.label(output)~=0) 
    while(imdb.images.label(output)==0)  %if val/test, sample again
        output = index(randi(numel(index)));
    end
end
