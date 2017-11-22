function str = model_description_str(learn_type, mm_type, numClusters, rep_type, dim_red_type, instance_postfix)

	if isempty(learn_type)
		learn_type_prefix = '';
	else
		learn_type_prefix = [learn_type '_'];
    end
    
    % in the baselines, numClusters is irreleavnt
    if ~sts_type_is_baseline(mm_type)
        mm_type = sprintf('%s_%d', mm_type, numClusters);
    end
		
    str = sprintf('%s%s_%s%s%s', learn_type_prefix, mm_type, rep_type, postfix(dim_red_type), instance_postfix);
end
