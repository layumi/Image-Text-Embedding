clear;
netStruct = load('./data/res52_batch16_new_hope_word_fixed_jitter/net-epoch-120.mat');
net = dagnn.DagNN.loadobj(netStruct.net);
clear netStruct;
net.mode = 'test' ;
net.move('gpu') ;
%net.removeLayer('RankLoss');
net.conserveMemory = true;
im_mean = reshape(net.meta.normalization.averageImage,1,1,3);

load('./url_data.mat');
p = imdb.images.data(imdb.images.set==3);
ff = [];
%%------------------------------

for i = 1:1000
    disp(i);
    str = [ p{i}];
    im = imread(str);
    oim = im;   % or oim = im;
    f = getFeature2(net,oim,im_mean,'data','fc1_1bn');
    f = sum(sum(f,1),2);
    f2 = getFeature2(net,fliplr(oim),im_mean,'data','fc1_1bn');
    f2 = sum(sum(f2,1),2);
    f = f+f2;
    size4 = size(f,4);
    f = reshape(f,[],size4)';
    f = norm_zzd(f);
    ff = cat(1,ff,f);
end
save('./test/resnet_flikr30k_pool5_img.mat','ff','-v7.3');
%}

ff = [];
load('./Flickr30k/dense_feature_word.mat');
imdb.charcnn = reshape(wordcnn,1,1,20074,[]); 
test_set = find(imdb.images.set==3);
for i = 1:1000
    disp(i);
    for j=1:5
        txt = imdb.charcnn(:,:,:,test_set(i)*5-5+j);
        f = getFeature2(net,txt,[],'data2','fc5_2bn');
        size4 = size(f,4);
        f = reshape(f,[],size4)';
        f = norm_zzd(f);
        ff = cat(1,ff,f);
    end
end
save('./test/resnet_flikr30k_pool5_txt.mat','ff','-v7.3');
