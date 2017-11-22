function k = get_next_char_index(str, i, c)

    k = i;
    while str(k) ~= c
        k = k + 1;
    end
    
end
