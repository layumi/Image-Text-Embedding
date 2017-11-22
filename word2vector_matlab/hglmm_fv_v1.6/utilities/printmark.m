function printmark(i, resolution, num_iterations)
    if ~mod(i, resolution)
        fprintf('.');
    end
    
    newline_thresh = 50*resolution;
    
    if ~mod(i, newline_thresh)
        fprintf(' %d\n', i);
    end

    if (i == num_iterations)
        fprintf(' %d\n', i);
    end
    
end
