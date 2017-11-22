% interval_sec: interval for polling (in seconds)
% timeout_intervals: timeout, in granularity of interval as defined by
% interval_sec. if timeout_intervals is not passed then no timeout.
% ret_val: 1 if the file was deteced, 0 if timeout
function ret_val = wait_for_file(file_name, interval_sec, timeout_intervals)

    if ~exist('timeout_intervals', 'var')
        timeout_intervals = Inf;
    end

    count = 0;
    
    while ~exist(file_name, 'file')
        pause(interval_sec);
        count = count + 1;
        
        if count == timeout_intervals
            ret_val = 0;
            return;
        end
    end

    ret_val = 1;
end
