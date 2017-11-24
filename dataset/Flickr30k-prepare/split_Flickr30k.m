% I follow the test set in Flickr8k.
test_fid = fopen('./Flickr8k_text/Flickr_8k.testImages.txt');
val_fid = fopen('./Flickr8k_text/Flickr_8k.devImages.txt');

test_name = textscan(test_fid,'%s');
val_name = textscan(val_fid,'%s');

file = dir('flickr30k-images/*.jpg');

set = ones(numel(file),1,'single');
parfor i = 1:numel(file)
   name = file(i).name(1:end-4);
   disp(i);
   is_val = cellfun(@(x) strcmp(x(1:end-15),name),val_name{1});
   is_test = cellfun(@(x) strcmp(x(1:end-15),name),test_name{1});
   
   if(  sum(is_val)>0 )
       set(i) = 2;
   end
   if(  sum(is_test)>0  )
       set(i) = 3;
   end
   %}
    
end

save('train_val_test_split.mat','set');
