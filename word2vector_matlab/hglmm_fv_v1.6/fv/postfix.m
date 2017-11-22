function result = postfix(str)
    if isempty(str)
        result = '';
    else
        result = sprintf('_%s', str);
    end
end
