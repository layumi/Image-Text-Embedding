clear;
load('../dataset/Flickr30k-prepare/url_data.mat');
load('../dataset/Flickr30k-prepare/train_val_test_split.mat');
load('../dataset/Flickr30k-prepare/rawdata-txt.mat');
% get original txt gallery  (size==5000)
test_set = find(set==3);

% get image feature
ff1 = load('./resnet_flikr30k_pool5_img.mat');
ff2 = load('./resnet_flikr30k_pool5_txt.mat');
%raw = imdb.images.txt(1:5:end);
%raw = raw(test_set);
%rank 

raw_test = [];
for i=1:1000
    tmp = test_set(i)*5-4 : test_set(i)*5;
    raw_test = cat(1,raw_test,raw_txt(tmp));
end

p = dir('./test-image-for-show/*.jpg');
for i = 1:size(ff1.ff,1)
    %disp(i);
    %title(raw{i});
    tmp = ff1.ff(i,:);
    score = tmp*(ff2.ff)';
    [s, index] = sort(score, 'descend');    
    good_index = (i*5-4):i*5;
    %{
    im = imread(['./test-image-for-show/' p(i).name]);
    %imshow(im);
    imwrite(im,sprintf('./result-txt-for-show/%d.jpg',i));
    fid = fopen(sprintf('./result-txt-for-show/%d.txt',i),'w');
    fprintf(fid,'ground truth\n');
    for ii=1:5
       fprintf(fid,'%s\n',raw_test{(good_index(ii))});
    end    
    fprintf(fid,'our result\n');
    for ii=1:5
       fprintf(fid,'%s\n',raw_test{(index(ii))});
    end
    fclose(fid);
    %}    
    junk_index = []; 
    [ap(i), CMC(i, :)] = compute_AP_short(good_index, junk_index, index);
end

CMC = mean(CMC);
rank = find(CMC>0.5);
fprintf('image-txt rank-1:%f mAP:%f Medr:%f\n', CMC(1),mean(ap),rank(1));
fprintf('image-txt rank-5:%f rank-10:%f\n', CMC(5),CMC(10));

clear CMC;
clear ap;
for i = 1:size(ff2.ff,1)
    %disp(i);
    %imshow(imread(['./test-image-for-show/' p(i).name]));
    %title(raw{i});
    tmp = ff2.ff(i,:);
    score = tmp*(ff1.ff)';
    [s, index] = sort(score, 'descend');
    good_index = floor((i-1)/5)+1;
    %{
    fid = fopen(sprintf('./result-img-for-show/%d.txt',i),'w');
    fprintf(fid,'%s\n',raw_test{i});
    im = imread(['./test-image-for-show/' p(good_index).name]);
    imwrite(im,sprintf('./result-img-for-show/%d_good.jpg',i));
    for ii=1:5
        im = imread(['./test-image-for-show/' p(index(ii)).name]);
        imwrite(im,sprintf('./result-img-for-show/%d_%d.jpg',i,ii));
    end
    %}
    %title(raw_test{(index(1))});
    junk_index = []; 
    [ap(i), CMC(i, :)] = compute_AP_short(good_index, junk_index, index);
end
CMC = mean(CMC);
rank = find(CMC>0.5);
fprintf('txt-image rank-1:%f mAP:%f Medr:%f\n', CMC(1),mean(ap),rank(1));
fprintf('txt-image rank-5:%f rank-10:%f\n', CMC(5),CMC(10));
