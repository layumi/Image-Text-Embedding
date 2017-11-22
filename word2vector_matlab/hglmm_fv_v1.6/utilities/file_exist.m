function result = file_exist(file_name)

    result = (exist(file_name, 'file') == 2);

end
