% mm_type: 'gmm_10', 'lmm_10', ...
function mm_file_name = get_mm_file_name(mm_type, dim_red_type, is_sampled)

    if is_sampled
        sampled_str = '_sampled';
    else
        sampled_str = '';
    end
    
    mm_file_name = add_data_dir_base(sprintf('GoogleNews_norm_%s%s%s.mat', mm_type, sampled_str, postfix(dim_red_type)));

end
