% resize image to 256*256
% and calculate the image mean
mkdir imgs_256
p = dir('./imgs/');
m1 = [];
for i = 3:numel(p)  %remove . & ..
    str = [ './imgs/',p(i).name,'/*.jpg'];
    pp = dir(str);
    if(numel(pp)==0)
        str = [ './imgs/',p(i).name,'/*.bmp'];
        pp = dir(str);
    end
    if(numel(pp)==0)
        str = [ './imgs/',p(i).name,'/*.png'];
        pp = dir(str);
    end
    for j = 1:numel(pp)
        img_str = [ './imgs/',p(i).name,'/',pp(j).name];
        wimg_str = [ './imgs_256/',p(i).name,'/',pp(j).name(1:end-3),'jpg'];
        wdir_str = ['./imgs_256/',p(i).name];
        mkdir(wdir_str);
        im = imresize(imread(img_str),[256,256]);
        imwrite(im,wimg_str);
        m = mean(mean(im,1),2);
        m1 = cat(4,m1,m);
    end
end

mm = mean(m1,4);