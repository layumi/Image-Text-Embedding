clear;
netStruct = load('../data/res52_coco_batch32_Rankloss_2:1:0.1_margin1_both_drop0.5_hard_256/net-epoch-20.mat');
net = dagnn.DagNN.loadobj(netStruct.net);
clear netStruct;
net.mode = 'test' ;
net.move('gpu') ;
net.removeLayer('RankLoss');
net.conserveMemory = true;
im_mean = reshape(net.meta.normalization.averageImage,1,1,3);

load('../dataset/MSCOCO-prepare/url_data.mat');
p = imdb.images.data(imdb.images.set==3);
ff = [];
%%------------------------------

for i = 1:numel(p)
    disp(i);
    str = p{i};
    im = imread(str);
    oim = im; %imresize(im,[224,224]);
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
save('./resnet_coco_img.mat','ff','-v7.3');
%}

ff = [];
load('../dataset/MSCOCO-prepare/coco_word2.mat');
load('../dataset/MSCOCO-prepare/test_id.mat')
test_word = wordcnn(:,test_id.txt_id>0);

for i = 1:size(test_word,2)
    disp(i);
    %get vector
    content = test_word(:,i);
    len = numel(find(content>0));
    txtinput = zeros(len,29972,'single');
    for k=1:len %32
        txtinput(k,content(k))=1;
    end
    win = 33-len;
    input = zeros(32,29972,win,'single');
    for kk = 1:win
        input(kk:kk+len-1,:,kk) = txtinput;
    end
    input = reshape(input,1,32,29972,[]);
    f = getFeature2(net,input,[],'data2','fc5_2bn');
    f = sum(f,4);
    size4 = size(f,4);
    f = reshape(f,[],size4)';
    f = norm_zzd(f);
    ff = cat(1,ff,f);
end
save('./resnet_coco_txt.mat','ff','-v7.3');

evaluate1K;
evaluate5K;
