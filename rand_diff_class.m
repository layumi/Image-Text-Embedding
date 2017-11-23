function output = rand_diff_class(imdb,index)
%only for flick30k
    output = randi(numel(imdb.images.data));
    while( output == index || imdb.images.set(output)~=1)
        output = randi(numel(imdb.images.data)); 
    end
    output = (output-1)*5+randi(5);
end