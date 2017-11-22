% the function assumes there is only one number in the string!
function num = str_extract_double(str)
    num_str = str( ismember(str, '.0123456789') );
	num = str2double(num_str);
end
