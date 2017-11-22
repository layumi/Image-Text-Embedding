% cword is the cell arrary containing the words
% M is a matrix whose columns are the word2vec vectors of the words
function [cword,M] = read_word2vec_binfile(binfile, sqrt_flag, norm_flag)

% set the character encoding if needed
% (there is a problem reading the word2vec binfile when encoding is UTF-8)
cur_encoding = slCharacterEncoding();
needed_encoding = 'ISO-8859-1';
if ~strcmp(cur_encoding, needed_encoding)
  slCharacterEncoding(needed_encoding);
  lg(1, 'character encoding was set to: %s\n', needed_encoding);
end

if ~exist('binfile','var')
  binfile = 'outputvecs';
elseif binfile(end)=='/'
  binfile = fullfile(binfile,'outputvecs');
end
f = fopen(binfile,'r');


thewords = fscanf(f,'%d',1);
thesize =  fscanf(f,'%d',1);
%fgetl(f)
%fscanf(f,'%c',4)
M = zeros(thesize,thewords);
cword = cell(1, thewords);

lg(1, 'num of words  : %d\n', thewords);
lg(1, 'dimensionality: %d\n', thesize);

tic

for i = 1:thewords,
  if mod(i, 100000) == 0
      lg(1, 'i = %d\n', i);
  end
  
  cword{i} = fscanf(f,'%s',1);
  fscanf(f,'%c',1);
  M(:,i) = fread(f,thesize,'float');
end
fclose(f);

toc

if sqrt_flag
    lg(1, 'sqrt...\n');
    
	for i = 1:thewords,
	  if mod(i, 100000) == 0
	      lg(1, 'i = %d\n', i);
      end
      
	  M(:,i) = sign(M(:,i)) .* sqrt(abs(M(:,i)));
	end
end

if norm_flag
    lg(1, 'normalize...\n');
    M = normc(M);
end
