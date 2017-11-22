function words = allwords(str,sepchars)
% allwords: converts a string into a cellstring of separate words
% usage: words = allwords(str)
% usage: words = allwords(str,sepchars)
%
%
% arguments: (input)
%  str - any vector string. str may also be a vector of
%        another class that contains integers.
%
%  sepchars - (OPTIONAL) separator characters that
%        will be used to parse the string into
%        separate words.
%
%        If str is a character string, then sepchars
%        by default will be ' .,;:?!', together with
%        the tab and carriage return characters. The
%        latter are ascii characters 9 and 13.
% 
%        If str is a numeric (integer) vector, then
%        the default for sepchars will be [-inf inf nan].
%
%        Consecutive separator characters will be
%        treated as a single such character.
%
% arguments: (output)
%  word - a cell array of strings, one cell for each
%        distinct word as separated by the supplied
%        separator characters.
%
%
% Example:
% % Parse a sentence. One or more spaces and punctuation
% % characters are all valid separator characters.
% str = 'The    quick brown fox jumped  over the lazy dog.';
% allwords(str)
% % ans = 
% %  'The' 'quick' 'brown' 'fox' 'jumped' 'over' 'the' 'lazy' 'dog'
%
% Example:
% % Parse a string of integers, with only NaN
% % elements as a separator.
% str = [1 2 4 2 inf 3 3 5 nan 4 6 5];
% words = allwords(str,nan);
% words{:}
% % ans =
% %     1     2     4     2   Inf     3     3     5
% % ans =
% %     4     6     5
%
% Example:
% % allwords is fast, here on a random string of length 1e6.
% str = round(rand(1,1000000)*10);
% tic
% words = allwords(str,[0 10]);
% toc
% % Elapsed time is 0.455194 seconds.
% 
% % There were over 90000 different words that were extracted
% numel(words)
% % ans =
% %     90310
%
% % The longest word had length
% max(cellfun(@numel,words))
% % ans =
% %     104
%
%
% See also: strtok, cellstr, strfun
%
% Author: John D'Errico
% e-mail: woodchips@rochester.rr.com
% Release date: 4/7/2010

% test for problems.
% first, verify that str is a vector.
if ~isvector(str)
  error('ALLWORDS:nonvectorinput','str must be a vector')
end
if nargin > 2
  error('ALLWORDS:toomanyargs','No more than two arguments allowed')
elseif nargin < 1
  help allwords
  return
end

if (nargin < 2) || isempty(sepchars)
  if ischar(str)
    % this is a character string. The default for
    % sepchars will be the standard sentence
    % punctuation characters, as well as white space.
    sepchars = [' .,;:?!',char([9 13])];
  elseif all((round(str) == str) | isnan(str))
    % the numeric sepchars are simple...
    sepchars = [-inf inf nan];
  else
    % I don't know what to do with this input class
    error('ALLWORDS:improperarg', ...
      'str must be character or integer of some class')
  end
end

% empty begets empty
if isempty(str)
  words = {};
  return
end

% just in case, is it possible that someone
% might give an ARRAY of sepchars? 
sepchars = sepchars(:);

% ensure that str is a row vector
str = reshape(str,1,[]);

% ensure that the first and last characters
% are separators
if ~ismember(str(1),sepchars)
  str = [sepchars(1),str];
end
if ~ismember(str(end),sepchars)
  str = [str,sepchars(1)];
end

% locate every separator char
scind = ismember(str,sepchars);
if any(isnan(sepchars))
  % a nan was in the list of separator characters,
  % but ismember won't find nans.
  scind = scind | isnan(str);
end

% find the location of the beginning of each word
wordbegin = strfind(scind,[1 0]);
wordend = strfind(scind,[0 1]);

% clearly, there must be a beginning to each word
% that precedes the end of that word. So the length
% of those words is just this difference.
wordlengths = wordend - wordbegin;

% how many words did we find?
nwords = numel(wordlengths);
words = cell(1,nwords);

% extract the words into a cell array
[uniklengths,I,J] = unique(wordlengths); %#ok
for ulen = uniklengths
  k = find(wordlengths == ulen);
  
  % extract each word of length exactly ulen
  ind = bsxfun(@plus,wordbegin(k).',1:ulen);
  
  % and stuff those strings into the corresponding cell
  if ulen == 1
    % be careful with substrings of length 1
    if ischar(str)
      % we can gain by special casing the character string
      % case. cellstr is faster for them than mat2cell.
      words(k) = cellstr(str(ind).');
    else
      words(k) = mat2cell(str(ind).',ones(numel(k),1),ulen);
    end
  else
    if ischar(str)
      words(k) = cellstr(str(ind));
    else
      words(k) = mat2cell(str(ind),ones(numel(k),1),ulen);
    end
  end
end


