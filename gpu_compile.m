addpath matlab;
addpath test;
addpath examples;
run matlab/vl_setupnn ;
%{
vl_compilenn('enableGpu', true, ...
'cudaRoot', '/usr/local/cuda', ...  %change it 
'cudaMethod', 'nvcc',...
'enableCudnn',true,... 
'cudnnroot','/home/zzd/image-txt-retrieval/cuda8.0_cudnn5.1.10');
%}
warning('off');
