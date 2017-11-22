function [ap, cmc] = compute_AP(good_image, junk_image, index)

cmc = zeros(length(index), 1);
ngood = length(good_image);

%remove junk_image
rows_junk = ismember(index,junk_image);
index(rows_junk) = [];

%find good_index
rows_good = find(ismember(index,good_image));
cmc(rows_good(1):end) = 1; %after first equals to one

ap = 0;
for i = 1:ngood
    d_recall = 1/ngood;
    precision = i/rows_good(i);
    if (rows_good(i)~=1)
        old_precision = (i-1)/(rows_good(i)-1);
    else
        old_precision = 1;
    end
    ap = ap + d_recall*(old_precision+precision)/2;
end

end


