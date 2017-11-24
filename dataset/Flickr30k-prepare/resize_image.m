s = 256;
mkdir flickr30k-images-256
p = dir('./flickr30k-images/*.jpg');
wp = sprintf('./flickr30k-images-%d/',s);
h1 = [];
w1 = [];
for i = 1:numel(p)
    %disp(i);
    str = [ './flickr30k-images/',p(i).name];
    im = imread(str);
    [h,w,~] = size(im);
    h1 = cat(1,h1,h);
    w1 = cat(1,w1,w);
    
    if(h<w)
        rate = s/h;
        imr = imresize(im,[s,w*rate]);
    else
        rate = s/w;
        imr = imresize(im,[h*rate,s]);
    end
    imwrite(imr,[ wp,p(i).name]);
    %}
end

disp( mean(h1) );  % 395
disp( mean(w1) );  % 460