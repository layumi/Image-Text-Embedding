%code for supplementary material
clear;
netStruct = load('../data/res152_batch32_Rankloss_2:1:0.1_margin1_both_drop0.75_shift_hard_256_152x152/net-epoch-80.mat');
net = dagnn.DagNN.loadobj(netStruct.net);
clear netStruct;
net.mode = 'test' ;
net.move('gpu') ;
net.removeLayer('RankLoss');
net.conserveMemory = true;
im_mean = reshape(net.meta.normalization.averageImage,1,1,3);
load('../url_data.mat');
p = imdb.images.data(imdb.images.set==3);
%%-------image feature
which_img = 621;
str = [ p{which_img}];
im = imread(['.',str]);
imshow(im);
oim = im;   % or oim = im;
f = getFeature2(net,oim,im_mean,'data','fc1_1bn');
f = sum(sum(f,1),2);
f2 = getFeature2(net,fliplr(oim),im_mean,'data','fc1_1bn');
f2 = sum(sum(f2,1),2);
f = f+f2;
size4 = size(f,4);
f = reshape(f,[],size4)';
f_img = norm_zzd(f);

% get raw text
load('../Flickr30k/rawdata-txt.mat');
raw_test = [];
load('../Flickr30k/train_val_test_split.mat');
test_set = find(set==3);
for i=1:1000
    tmp = test_set(i)*5-4 : test_set(i)*5;
    raw_test = cat(1,raw_test,raw_txt(tmp));
end
which_sentence = which_img*5-3;
title(raw_test{which_sentence});
load('../Flickr30k/dense_feature_word2.1.mat');
test_set = find(imdb.images.set==3);

%get text feature
content = wordcnn(:,test_set(which_img)*5-3);
len = sum(content>0);
load('/home/zzd/nlp/word2vector_matlab/flickr30k_dictionary.mat');
word_name = subset.names;
for k=1:len
    fprintf('%s ',word_name{content(k)});
end
fprintf('\n');

for i = 0:len
    content_tmp = content;
    if(i~=0) %start block words
        content_tmp(i)=0;
    end
    txtinput = zeros(len,20074,'single');
    kk = 1;
    for k=1:32
        if(content_tmp(k)==0)
            continue;
        end
        txtinput(kk,content_tmp(k))=1;
        kk = kk+1;
    end
    %transfer it to different location
    win = 33-len;
    input = zeros(32,20074,win,'single');
    for kk = 1:win
        input(kk:kk+len-1,:,kk) = txtinput;
    end
    
    input = reshape(input,1,32,20074,[]);
    f = getFeature2(net,input,[],'data2','fc6_2bn');
    f = sum(f,4);
    size4 = size(f,4);
    f = reshape(f,[],size4)';
    f_txt = norm_zzd(f);
    
    score = f_img * f_txt';
    if(i==0)
        s0 = score;
    else
      fprintf('%s,%.4f\n',word_name{content(i)},score-s0);
    end
end