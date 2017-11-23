fprintf('---------------5K test-----------------\n');

clear;
load('../../MSCOCO/test_id.mat');
load('../../MSCOCO/caption.mat');
load('../../MSCOCO/url_data.mat');
% get image feature
ff1 = load('./resnet_coco_img.mat');
ff2 = load('./resnet_coco_txt.mat');

txt_id = test_id.txt_id;
test_caption = caption(txt_id>0);
txt_id(txt_id==0) = [];
img_id = test_id.img_id;
img_id(img_id==0) = [];
test_url = imdb.images.data(imdb.images.set==3);


% img query
for i = 1:size(ff1.ff,1)
    %disp(i);
    tmp = ff1.ff(i,:);
    score = tmp*(ff2.ff)';
    [s, index] = sort(score, 'descend');
    good_index = find(txt_id==img_id(i));
    junk_index = []; 
    %{
    im = imread(test_url{i});
    imwrite(im,sprintf('./result-txt-for-show/%d.jpg',i));
    fid = fopen(sprintf('./result-txt-for-show/%d.txt',i),'w');
    fprintf(fid,'ground truth\n');
    for ii = 1:numel(good_index)
        fprintf(fid,'%s\n',test_caption{good_index(ii)});
    end
        fprintf(fid,'our result\n');
    for ii = 1:5
        fprintf(fid,'%s\n',test_caption{index(ii)});
    end
    fclose(fid);
    %}
    [ap(i), CMC(i, :)] = compute_AP_rerank(good_index, junk_index, index);
end
CMC = mean(CMC);
rank = find(CMC>0.5);
fprintf('image-txt rank-1:%f mAP:%f Medr:%f\n', CMC(1),mean(ap),rank(1));
fprintf('image-txt rank-5:%f rank-10:%f\n', CMC(5),CMC(10));

ap = [];
CMC = [];
% txt query
parfor i = 1:size(ff2.ff,1)
    %disp(i);
    tmp = ff2.ff(i,:);
    score = tmp*(ff1.ff)';
    [s, index] = sort(score, 'descend');
    good_index = find(img_id==txt_id(i));
    %query_title = test_caption(i);
    junk_index = []; 
    [ap(i), CMC(i, :)] = compute_AP_rerank(good_index, junk_index, index);
end
CMC = mean(CMC);
rank = find(CMC>0.5);
fprintf('txt-image rank-1:%f mAP:%f Medr:%f\n', CMC(1),mean(ap),rank(1));
fprintf('txt-image rank-5:%f rank-10:%f\n', CMC(5),CMC(10));