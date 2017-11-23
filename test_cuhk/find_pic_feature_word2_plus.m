clear;
netStruct = load('../data/res52_cuhk_batch32_Rankloss_2:1:0.5_margin1/net-epoch-60.mat');
net = dagnn.DagNN.loadobj(netStruct.net);
clear netStruct;
net.mode = 'test' ;
net.move('gpu') ;
net.removeLayer('RankLoss');
net.conserveMemory = true;
im_mean = reshape(net.meta.normalization.averageImage,1,1,3);

load('/home/zzd/CUHK-PEDES/url_data.mat');
p = imdb.images.data(imdb.images.set==3);
%%----extract img

for j = 1:3:300  % loop for different img
    which_img = j;
    str = p{which_img};
    im = imread(str);
    imshow(im);
    oim = im;
    imwrite(im,sprintf('./attention-text-for-show/%d.jpg',which_img));
    f = getFeature2(net,oim,im_mean,'data','fc1_1bn');
    f = sum(sum(f,1),2);
    f2 = getFeature2(net,fliplr(oim),im_mean,'data','fc1_1bn');
    f2 = sum(sum(f2,1),2);
    f = f+f2;
    size4 = size(f,4);
    f = reshape(f,[],size4)';
    f_img = norm_zzd(f);
    
    % extract
    load('/home/zzd/CUHK-PEDES/test_id.mat');
    txt_id = test_id.txt_id;
    img_id = test_id.img_id;
    which_text = find(txt_id == img_id(which_img));
    which_text = which_text(1);
    
    load('/home/zzd/CUHK-PEDES/cuhk_word2.mat');
    load('//home/zzd/CUHK-PEDES/CUHK-PEDES_dictionary.mat');
    word_name = subset.names;
    wordcnn = wordcnn(:,end-6155:end);
    content = wordcnn(:,which_text);
    len = numel(find(content>0));
    for k = 1:len
        fprintf('%s ',word_name{content(k)});
    end
    fprintf('\n');
    for i = 0:len
        content_tmp = content;
        if(i~=0)
            content_tmp(i) = 0;
        end
        txtinput = zeros(len,7263,'single');
        kk = 1;
        for k=1:56
            if(content_tmp(k)==0)
                continue;
            end
            txtinput(kk,content_tmp(k))=1;
            kk = kk+1;
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
        f_txt = norm_zzd(f);
        
        score = f_img * f_txt';
        if(i==0)
            s0 = score;
        else
            fid = fopen(sprintf('./attention-text-for-show/%d.txt',which_img),'a');
            fprintf(fid,'%s : %.4f\n',word_name{content(i)},score-s0);
            fclose(fid);
        end
    end
end