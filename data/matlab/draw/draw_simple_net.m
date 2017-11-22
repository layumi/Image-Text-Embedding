function draw_simple_net(net,name)
%input: net (DAGNN format)  output_filename output: jpg
%This code is modified from http://stackoverflow.com/questions/5518200/automatically-generating-a-diagram-of-function-calls-in-matlab
OutputName = 'net';
if ~isempty(name)
    OutputName = name;
end
dotFile = [OutputName '.dot'];

% Render to image
imageFile = [OutputName '.png'];
% Assumes the GraphViz bin dir is on the path; if not, use full path to dot.exe
cmd = sprintf('dot -Tpng -Gsize="48,48" "%s" -o"%s"', dotFile, imageFile);  % for better view, you can use number bigger than 32
system(cmd);
fprintf('Wrote to %s\n', imageFile);
%im = imread(imageFile);
%imshow(im);