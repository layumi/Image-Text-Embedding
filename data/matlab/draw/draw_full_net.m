function draw_full_net(net,name)
%input: net (DAGNN format)  output_filename output: jpg
%This code is modified from http://stackoverflow.com/questions/5518200/automatically-generating-a-diagram-of-function-calls-in-matlab
OutputName = 'net';
if ~isempty(name)
    OutputName = name;
end
dotFile = [OutputName '.dot'];
fid = fopen(dotFile, 'w');
fprintf(fid, 'digraph G {\n');
fprintf(fid,'rankdir=LR;\n');
for i = 1:numel(net.layers)
    for j = 1:numel(net.layers(i).inputs)  % multi-input to one output
        fprintf( fid,'%s -> %s', char(net.layers(i).inputs(j)), char(net.layers(i).outputs));
        block = net.layers(i).block;
        block_class = class(block);
        if (isequal(block_class ,'dagnn.Conv'))
            fh = block.size(1);
            fw = block.size(2);
            fcount=block.size(4);
            step=block.stride(1);
            fprintf (fid,'[label = %s%dx%dx%dStride%d];\n',block_class(7:end),fh,fw,fcount,step);
        elseif isequal(block_class ,'dagnn.Pooling')
            fh = block.poolSize(1);
            fw = block.poolSize(2);
            step=block.stride(1);
            fprintf (fid,'[label = %s%dx%dStride%d];\n',block_class(7:end),fh,fw,step);
       % elseif isequal(block_class ,'dagnn.DropOut')
           % rate = block.rate;
            %fprintf (fid,'[label = %s%.1f];\n',block_class(7:end),rate);
        else
            fprintf (fid,'[label = %s];\n',block_class(7:end));
        end
    end
end
fprintf(fid, '}\n');
fclose(fid);

% Render to image
imageFile = [OutputName '.png'];
% Assumes the GraphViz bin dir is on the path; if not, use full path to dot.exe
cmd = sprintf('dot -Tpng -Gsize="2048,2048" "%s" -o"%s"', dotFile, imageFile);  % for better view, you can use number bigger than 32
system(cmd);
fprintf('Wrote to %s\n', imageFile);
%im = imread(imageFile);
%imshow(im);