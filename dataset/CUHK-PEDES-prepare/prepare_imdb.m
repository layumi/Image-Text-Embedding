%txt = fileread('caption_all.json');
txt = fileread('reid_raw.json');
json = jsondecode(txt);  %jsondecode is not available on 2015b. I use Matlab2016b.

file_path = {json.file_path};
caption = {json.captions};
id = [json.id];
% final 2000 id as val and test, as in the offical split.
set = [ones(1,34054),2*ones(1,3078),3*ones(1,3074)]; 

% dataset
imdb.images.set = set;
% change to full path
imdb.images.data = cellfun(@(x) sprintf('/home/zzd/Image-Text-Embedding/dataset/CUHK-PEDES-prepare/imgs_256/%sjpg',x(1:end-3)), file_path,'UniformOutput',false);;
id(set~=1) = 0;  % remove val and test id for training
imdb.images.label = id;
count =1;
for i=1:numel(json)
    len = numel(json(i).captions);
    label2(count:count+len-1) = id(i);
    count = count+len;
end
imdb.images.label2 = label2;
%resize_image; % resize image to 256x256, return mm.
%imdb.rgbMean = mm;
save('url_data.mat','imdb');  % only train data

% make dictionary
caption_train = caption(set==1);
caption_dic = [];
for i = 1:numel(caption_train)
    caption_dic = cat(1,caption_dic,caption_train{i});
end

save('caption_train.mat','caption_dic');


for i = numel(caption_train)+1:numel(caption)
    caption_dic = cat(1,caption_dic,caption{i});
end

save('caption.mat','caption_dic');
