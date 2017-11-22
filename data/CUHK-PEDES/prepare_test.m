txt = fileread('reid_raw.json');
json = jsondecode(txt);  %jsondecode is not available on 2015b. I use Matlab2016b.

json = json(end-3073:end);  %test

img_id = [json.id];
txt_id = [];

for i = 1:numel(img_id)
    num = numel(json(i).captions);
    txt_id = cat(1,txt_id,repmat(img_id(i),num,1));
end

test_id.img_id = img_id;
test_id.txt_id = txt_id;

save('test_id.mat','test_id');