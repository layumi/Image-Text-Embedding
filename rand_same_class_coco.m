function output = rand_same_class_coco(imdb,label)
    location = (label+10000)*5;
    start = location-1000;
    last = min(location+2000,numel(imdb.images.label2));
    quick = imdb.images.label2(start:last);
    index = find(quick==label);
    output = start -1 + index(randi(numel(index)));
end