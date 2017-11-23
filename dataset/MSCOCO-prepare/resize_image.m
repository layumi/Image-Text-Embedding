mkdir image_256
p = dir('./image/');
m1 = [];
for i = 3:numel(p)  %remove . & ..
    str = [ './image/',p(i).name,'/*.jpg'];
    pp = dir(str);
    if(numel(pp)==0)
        str = [ './image/',p(i).name,'/*.bmp'];
        pp = dir(str);
    end
    if(numel(pp)==0)
        str = [ './image/',p(i).name,'/*.png'];
        pp = dir(str);
    end
    for j = 1:numel(pp)
        img_str = [ './image/',p(i).name,'/',pp(j).name];
        wimg_str = [ './image_256/',p(i).name,'/',pp(j).name(1:end-3),'jpg'];
        wdir_str = ['./image_256/',p(i).name];
        mkdir(wdir_str);
        im = imread(img_str);
        sz = size(im);
        if(numel(sz)==2)
           im = repmat(im,1,1,3); 
        end
        if(sz(1)<sz(2))
           im = imresize(im,[256, round(sz(2)/sz(1)*256)]);
        else
           im = imresize(im,[round(sz(1)/sz(2)*256),256]);
        end
        imwrite(im,wimg_str);
        m = mean(mean(im,1),2);
        m1 = cat(4,m1,m);
    end
end

mm = mean(m1,4);