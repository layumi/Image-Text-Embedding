fid = fopen('flickr30k-dense+num.txt');
% uid sid tweet
tline = fgetl(fid);
raw_txt = [];
count = 1;
ff = single(zeros(150,36,158915));
while ischar(tline)
    disp(count);
    fx = zeros(36,150,'single');
    f = tline-97+1; %get 1-26
    %num
    f(f<0) = f(f<0)+48+27;
    x = 0:numel(f)-1;
    x = x*36; % get offset
    f = f+x;
    fx(f)=1;
    ff(:,:,count) = fx';
    count = count + 1;
    tline = fgetl(fid);
end
fclose(fid);
save('dense_feature+num.mat','ff');