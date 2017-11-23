function [img_r1,img_r5,img_r10,img_med, img_map, txt_r1,txt_r5,txt_r10,txt_med,txt_map] = evaluate( ff1, ff2 ,img_id, txt_id)

parfor i = 1:size(ff1,1)
    %disp(i);
    tmp = ff1(i,:);
    score = tmp*(ff2)';
    [s, index] = sort(score, 'descend');
    good_index = find(txt_id==img_id(i));
    junk_index = []; 
    [ap(i), CMC(i, :)] = compute_AP_rerank(good_index, junk_index, index);
end
CMC = mean(CMC);
rank = find(CMC>0.5);
img_r1 = CMC(1); img_map = mean(ap); img_med=rank(1);
img_r5 = CMC(5); img_r10 = CMC(10);

ap = [];
CMC = [];
% txt query
parfor i = 1:size(ff2,1)
    %disp(i);
    tmp = ff2(i,:);
    score = tmp*(ff1)';
    [s, index] = sort(score, 'descend');
    good_index = find(img_id==txt_id(i));
    %query_title = test_caption(i);
    junk_index = []; 
    [ap(i), CMC(i, :)] = compute_AP_rerank(good_index, junk_index, index);
end
CMC = mean(CMC);
rank = find(CMC>0.5);
txt_r1 = CMC(1); txt_map = mean(ap); txt_med=rank(1);
txt_r5 = CMC(5); txt_r10 = CMC(10);

end

