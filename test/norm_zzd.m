function ff = norm_zzd(f)
% dim2 is feature dim
% for example:   2371 * 1024
s = sqrt(sum(f.^2,2));
dim = size(f,2);
s = repmat(s,1,dim);
ff = f./s;

end