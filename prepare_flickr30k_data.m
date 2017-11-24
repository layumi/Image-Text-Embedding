addpath ./examples/imagenet;
load('./dataset/Flickr30k-prepare/train_val_test_split.mat') ;
imdb.imageDir = './dataset/Flickr30k-prepare/flickr30k-images-256/';

filename = dir([imdb.imageDir,'*jpg']);
cell_data = {filename(1:end).name}.';
full_path = cellfun(@(x) [imdb.imageDir,x],cell_data,'UniformOutput',false);
imdb.images.data = full_path;
imdb.images.set = set;

count = 0;
for i=1:numel(set)
    if(set(i)==1)
        count = count +1;
        imdb.images.label(i) = single(count);
    else
        imdb.images.label(i) = 0;
    end
end

% Compute image statistics (mean, RGB covariances, etc.)
train = find(imdb.images.set == 1) ;
images = fullfile(imdb.images.data(train(1:end))) ;
[averageImage, rgbMean, rgbCovariance] = getImageStats(images, ...
    'imageSize', [256 256], ...
    'numThreads', 16, ...
    'gpus', 3) ;
%imdb.averageImage = averageImage;
imdb.rgbMean = rgbMean;
imdb.rgbCovariance =rgbCovariance;

save('url_data.mat','imdb');
