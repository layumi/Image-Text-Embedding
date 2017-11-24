clear;  
%Note that I add original val to train
%and then redefine the train/test split. 
%5000 val, 5000 test  https://cs.stanford.edu/people/karpathy/cvpr2015.pdf.
txt = fileread('./annotations/captions_val2014.json');
json = jsondecode(txt);  %jsondecode is not available on 2015b. I use Matlab2016b.

file_path_val = {json.images.file_name};
img_id_val = [json.images.id];

caption_val = {json.annotations.caption};
caption_img_id_val = [json.annotations.image_id];

%train
txt = fileread('./annotations/captions_train2014.json');
json = jsondecode(txt);  %jsondecode is not available on 2015b. I use Matlab2016b.

file_path = {json.images.file_name};
img_id = [json.images.id];

caption = {json.annotations.caption};
caption_img_id = [json.annotations.image_id];


data = cat(2,...
    cellfun(@(x) sprintf('./image_256/train2014/%sjpg',x(1:end-3)), file_path,'UniformOutput',false),...
    cellfun(@(x) sprintf('./image_256/val2014/%sjpg',x(1:end-3)), file_path_val,'UniformOutput',false));
img_label = cat(2,img_id,img_id_val);
caption = cat(2,caption,caption_val);
caption_label = cat(2,caption_img_id,caption_img_id_val);

% final 5000 images as test
rng(1);
num = numel(data);
set = ones(1,num);
val_test_id = randperm(num,10000);
set(1,val_test_id) = 2; %val+test
test_id = val_test_id(randperm(10000,5000));
set(1,test_id) = 3; %test

%assign image label
new_label = zeros(1,numel(img_label));
new_label(set==1) = 1:sum(set==1);
train_img_label = img_label; 
train_img_label(set~=1) = 0;
%assign text label
for i=1:numel(caption_label)
    now = caption_label(i);
    index = find(train_img_label==now);
    if(isempty(index))
        caption_label(i) = 0;
    else
        caption_label(i) = new_label(index);%update
    end
end

% dataset
imdb.images.set = set;
imdb.images.data = data;
imdb.images.label = new_label;
imdb.images.label2 = caption_label;
%resize_image; % resize image to 256, return mean.
imdb.rgbMean = [119.8557, 113.9346, 103.9336];
save('url_data.mat','imdb');  % only train data

caption_train = caption(caption_label~=0);
caption_dic = caption_train;

save('caption_train.mat','caption_dic');

save('caption.mat','caption');



caption_label = cat(2,caption_img_id,caption_img_id_val);
img_label = cat(2,img_id,img_id_val);

test_img_label = img_label(set==3);
%assign text label 
parfor i=1:numel(caption_label)
    now = caption_label(i);
    index = find(test_img_label==now);
    if(isempty(index))
        caption_label(i) = 0;
    end
end
test_id = [];
test_id.txt_id = caption_label; % test left
test_label = img_label;
test_label(set~=3) = 0;
test_id.img_id = test_label;  

save('test_id.mat','test_id');
