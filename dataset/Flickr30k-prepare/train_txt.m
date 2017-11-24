fid = fopen('results_20130124.token');
fid2 = fopen('flickr30k-train&val.txt','w');
load('train_val_test_split.mat');
test_index = find(set==3);
% uid sid tweet
tline = fgetl(fid);
raw_txt = [];
count = 1;
while ischar(tline)
    %skip test
    if ~isempty(find(ceil(count/5)==test_index))
        disp(count);
        count = count+1;
        tline = fgetl(fid);
        continue;
    end
    split_tline = strsplit(tline);
    s1 = numel(split_tline{1})+1;
    tline = tline(s1+1:end);
    fprintf(fid2,'%s\n',tline);
    %disp(tline)
    raw_txt{count,1} = tline;
    tline = fgetl(fid);
    count = count+1;
end
fclose(fid);
fclose(fid2);
%save('rawdata-txt.mat','raw_txt');



