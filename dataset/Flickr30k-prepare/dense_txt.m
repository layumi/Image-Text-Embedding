fid = fopen('results_20130124.token');
fid2 = fopen('flickr30k-dense.txt','w');
% uid sid tweet
tline = fgetl(fid);
raw_txt = [];
count = 1;
while ischar(tline)
    %disp(tline)
    split_tline = strsplit(tline);
    s1 = numel(split_tline{1})+1;
    tline = lower(tline(s1+1:end));
    is_charnum = isstrprop(tline,'alpha'); %'alphanum');
    tline = tline(is_charnum);
    if(numel(tline)>150)  %cut
        tline = tline(1:150);
    end
    fprintf(fid2,'%s\n',tline);
    %disp(tline)
    raw_txt{count,1} = tline;
    tline = fgetl(fid);
    %ll(count) = numel(tline);
    count = count+1;
end
fclose(fid);
fclose(fid2);
%fprintf('mean char: %d', mean(ll));
%save('rawdata-txt.mat','raw_txt');



