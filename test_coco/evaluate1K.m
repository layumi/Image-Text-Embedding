fprintf('---------------1K test-----------------');

clear;
load('../dataset/MSCOCO-prepare/test_id.mat');
load('../dataset/MSCOCO-prepare/caption.mat');
load('../dataset/MSCOCO-prepare/url_data.mat');
% get image feature
ff1 = load('./resnet_coco_img.mat');
ff_img = ff1.ff;
ff2 = load('./resnet_coco_txt.mat');
ff_txt = ff2.ff;

txt_id = test_id.txt_id;
test_caption = caption(txt_id>0);
txt_id(txt_id==0) = [];
img_id = test_id.img_id;
img_id(img_id==0) = [];
test_url = imdb.images.data(imdb.images.set==3);

% first 1000 img query

for j= 1:5
    start = 1+(j-1)*1000;
    late = j*1000;
    img_index = start:late;
    txt_index = [];
    for i=1:numel(txt_id)
        if ~isempty(find(txt_id(i)==img_id(img_index)))
            txt_index = cat(2,txt_index,i);
        end
    end
    
    [img_r1(j),img_r5(j),img_r10(j),img_med(j), img_map(j), txt_r1(j),txt_r5(j),txt_r10(j),txt_med(j),txt_map(j)] = ...
        evaluate(ff_img(img_index,:),ff_txt(txt_index,:),img_id(img_index),txt_id(txt_index));
end

fprintf('\n');
fprintf('image-txt rank-1:%f mAP:%f Medr:%f\n', mean(img_r1),mean(img_map),mean(img_med));
fprintf('image-txt rank-5:%f rank-10:%f\n', mean(img_r5),mean(img_r10));

fprintf('txt-image rank-1:%f mAP:%f Medr:%f\n', mean(txt_r1),mean(txt_map),mean(txt_med));
fprintf('txt-image rank-5:%f rank-10:%f\n', mean(txt_r5),mean(txt_r10));

