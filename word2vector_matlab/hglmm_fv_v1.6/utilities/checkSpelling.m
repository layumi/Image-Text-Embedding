function suggestion = checkSpelling(word)
%CHECKSPELLING  uses MSWord to correct spelling
%   CHECKSPELLING(WORD) checks the spelling of WORD and returns spelling
%   suggestions as a cell array.  If WORD is spelled correctly, CHECKSPELLING 
%   will return empty brackets.  If no suggestions are found.  CHECKSPELLING
%   returns 'no suggestions'.  
%Start the Word ActiveX Server and check the spelling of WORD
h = actxserver('word.application');
h.Document.Add;
correct = h.CheckSpelling(word);
if correct
      suggestion = []; %return empty if spelled correctly
else
      %If incorrect and there are suggestions, return them in a cell array
      if h.GetSpellingSuggestions(word).count > 0
          count = h.GetSpellingSuggestions(word).count;
          for i = 1:count
              suggestion{i} = h.GetSpellingSuggestions(word).Item(i).get('name');
          end
      else
          %If incorrect but there are no suggestions, return this:
          suggestion = 'no suggestions';
      end
end
%Quit Word to release the server
h.Quit
