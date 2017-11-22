function new_name = fname_concat(name, str)
    
    split = regexp(name, '.mat', 'split');
    new_name = strcat(split{1}, str, '.mat');
end
