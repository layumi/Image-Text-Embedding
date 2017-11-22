function tline = getNonEmptyLine(fileID)

    tline = '';
    
    while isempty(tline)
        tline = fgetl(fileID);
    end

end
