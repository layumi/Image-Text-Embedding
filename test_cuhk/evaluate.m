clear;
load('../dataset/CUHK-PEDES-prepare/test_id.mat');
load('../dataset/CUHK-PEDES-prepare/caption.mat');
load('../dataset/CUHK-PEDES-prepare/url_data.mat');
% get image feature
ff1 = load('./resnet_cuhk_img.mat');
ff2 = load('./resnet_cuhk_txt.mat');

txt_id = test_id.txt_id;
img_id = test_id.img_id;
test_url = imdb.images.data(imdb.images.set==3);
test_caption = caption_dic(end-6155:end);
for i = 1:size(ff2.ff,1)
    %disp(i);
    tmp = ff2.ff(i,:);
    score = tmp*(ff1.ff)';
    [s, index] = sort(score, 'descend');
    good_index = find(img_id==txt_id(i));
    query_title = test_caption(i);
    %imshow(test_url{good_index(1)});
    %------------for------show----------
    %{
    a = char(query_title);
    wp = sprintf('./result/%d_%s',i,a(1:min(size(a,2),128)));
    mkdir(wp);

    
    for j = 1:10
        %subplot(1,10,j);
        
        %title(s(j));
        origin_im = test_url{index(j)};
        imwrite(imread(origin_im),sprintf('%s/%f.jpg',wp,s(j)));
    end
    %}
    junk_index = []; 
    [ap(i), CMC(i, :)] = compute_AP_rerank(good_index, junk_index, index);
end
CMC = mean(CMC);
rank = find(CMC>0.5);
fprintf('txt-image rank-1:%f mAP:%f Medr:%f\n', CMC(1),mean(ap),rank(1));
fprintf('txt-image rank-5:%f rank-10:%f\n', CMC(5),CMC(10));
