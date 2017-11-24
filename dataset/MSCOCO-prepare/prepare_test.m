%val
%Note that I follow the other paper use val to train
%redefine the train/test split
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
    cellfun(@(x) sprintf('/media/share/MSCOCO/image_256/train2014/%sjpg',x(1:end-3)), file_path,'UniformOutput',false),...
    cellfun(@(x) sprintf('/media/share/MSCOCO/image_256/val2014/%sjpg',x(1:end-3)), file_path_val,'UniformOutput',false));
img_label = cat(2,img_id,img_id_val);
caption = cat(2,caption,caption_val);
caption_label = cat(2,caption_img_id,caption_img_id_val);

% final 1000 id as test
set = [ones(1,numel(data)-1000),3*ones(1,1000)];

new_label = 1:numel(img_label);
test_caption = [];
%start from 1
for i=1:numel(caption_label)
    now = caption_label(i);
    index = find(img_label==now);
    if(index<122288)  % 122288~123287
        continue;
    end
    test_caption = cat(1,test_caption,index);
end

test_id.txt_id = test_caption;
test_id.img_id = new_label(end-999:end);

save('test_id.mat','test_id');