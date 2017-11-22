function lg(show_time, format, varargin)

if show_time
    fprintf('[%s] %s', datestr(now, 'dd|HH:MM:SS'), ...
        sprintf(format, varargin{:}));
else
    fprintf(format, varargin{:});
end
