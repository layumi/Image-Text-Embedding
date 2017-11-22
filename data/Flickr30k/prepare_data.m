addpath ./examples/imagenet;
%load('./Flickr30k/flickr30k-txt.mat') ;
load('./Flickr30k/train_val_test_split.mat') ;
imdb.imageDir = './Flickr30k/flickr30k-images-400/';
%load('./Flickr30k/rawdata-txt.mat');
%norm 1   the hello has been normalized to 2.
%hello = norm_zzd(hello);

%norm 2
%tmp = reshape(hello',4800,5,[]);
%label = reshape(sum(tmp,2),4800,[]);
%label = norm_zzd(label');

%rank test
%tt = hello(1,:);
%score = hello*tt';
%[~,rank] = sort(score,'descend');

filename = dir([imdb.imageDir,'*jpg']);
cell_data = {filename(1:end).name}.';
full_path = cellfun(@(x) [imdb.imageDir,x],cell_data,'UniformOutput',false);
imdb.images.data = full_path;
imdb.images.set = set;
%imdb.images.txt = raw_txt;
%imdb.images.data2 = hello;

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
    'imageSize', [320 320], ...
    'numThreads', 16, ...
    'gpus', 3) ;
%imdb.averageImage = averageImage;
imdb.rgbMean = rgbMean;
imdb.rgbCovariance =rgbCovariance;

save('url_data_400.mat','imdb');