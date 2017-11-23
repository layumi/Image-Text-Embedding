function output = rand_same_class(imdb,label)
    index = find(imdb.images.label2==label);
    output = index(randi(numel(index)));
end