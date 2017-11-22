% in LMM / HGLMM, need to multiply by 10 to avoid numerical issue
LMM_HGLMM_MULT_FACTOR = 10;

global init_flag;
if isempty(init_flag)
    
    % location of this script
    pathstr = fileparts( mfilename('fullpath') );

    addpath([pathstr '/../utilities']);
    addpath([pathstr '/../FastICA_25']);
    addpath([pathstr '/../cca']);


    if isunix
        addpath([pathstr '/LMM_linux']);
        addpath([pathstr '/HGLMM_linux']);
    else
        addpath([pathstr '/LMM_win']);
        addpath([pathstr '/HGLMM_win']);
    end

    run([pathstr '/../vlfeat-0.9.18/toolbox/vl_setup']);

    clear pathstr;
    
    
    init_flag = true;
end
