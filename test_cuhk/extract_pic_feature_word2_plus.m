clear;
netStruct = load('../data/res52_cuhk_batch32_Rankloss_2:1:0.5_margin1/net-epoch-60.mat');
net = dagnn.DagNN.loadobj(netStruct.net);
clear netStruct;
net.mode = 'test' ;
net.move('gpu') ;
net.removeLayer('RankLoss');
net.conserveMemory = true;
im_mean = reshape(net.meta.normalization.averageImage,1,1,3);

load('../dataset/CUHK-PEDES-prepare/url_data.mat');
p = imdb.images.data(imdb.images.set==3);
ff = [];
%%------------------------------

for i = 1:numel(p)
    disp(i);
    str = p{i};
    im = imread(str);
    oim = im;
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
save('./resnet_cuhk_img.mat','ff','-v7.3');
%}

ff = [];
load('../dataset/CUHK-PEDES-prepare/cuhk_word2.mat');
test_word = wordcnn(:,end-6155:end);

for i = 1:6156
    disp(i);
    content = test_word(:,i);
    len = numel(find(content>0));
    txtinput = zeros(len,7263,'single');
    for k=1:len 
        txtinput(k,content(k))=1;
    end
    %transfer it to different location
    win = 57-len;
    input = zeros(56,7263,win,'single');
    for kk = 1:win
        input(kk:kk+len-1,:,kk) = txtinput;
    end
    input = reshape(input,1,56,7263,[]);
    f = getFeature2(net,input,[],'data2','fc5_2bn');
    f = sum(f,4);
    size4 = size(f,4);
    f = reshape(f,[],size4)';
    f = norm_zzd(f);
    ff = cat(1,ff,f);
end
save('./resnet_cuhk_txt.mat','ff','-v7.3');

evaluate;
