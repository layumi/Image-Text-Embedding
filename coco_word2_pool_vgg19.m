function net = resnet52_new_hope()

%----------------------img cnn----------------------
netStruct = load('./data/imagenet-vgg-verydeep-19.mat') ;
net = dagnn.DagNN.loadobj(netStruct) ;
net.removeLayer('fc8');
net.removeLayer('relu7');
net.removeLayer('prob');

for i = 1:numel(net.params)
    if(mod(i,2)==0)
        net.params(i).learningRate= 0; %0.02;
    else net.params(i).learningRate= 0; %0.001;
    end
    net.params(i).weightDecay=0; %1;
end

net.params(1).learningRate = 0; %1e-5;

fc1Block = dagnn.Conv('size',[1 1 4096 4096],'hasBias',false,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('fc1',fc1Block,{'x40'},{'fc1'},{'fc1f'});
net.addLayer('fc1bn',dagnn.BatchNorm(),{'fc1'},{'fc1bn'},...
    {'fc1bn_w','fc1bn_b','fc1bn_m'});
net.addLayer('fc1x',dagnn.ReLU(),{'fc1bn'},{'fc1bnx'});
fc1Block = dagnn.Conv('size',[1 1 4096 2048],'hasBias',false,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('fc1_1',fc1Block,{'fc1bnx'},{'fc1_1'},{'fc1_1f'});
net.addLayer('fc1_1bn',dagnn.BatchNorm(),{'fc1_1'},{'fc1_1bn'},...
    {'fc1_1bn_w','fc1_1bn_b','fc1_1bn_m'});
net.addLayer('fc1_1x',dagnn.ReLU(),{'fc1_1bn'},{'fc1_1bnx'});
net.addLayer('dropout',dagnn.DropOut('rate',0.5),{'fc1_1bnx'},{'fc1_1bnxd'});


%----------------------char cnn----------------------
% input is 1*32*20074*16
fc2Block = dagnn.Conv('size',[1 1 29972 300],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('fc2',fc2Block,{'data2'},{'fc2'},{'fc2f','fc2b'});
net.addLayer('fc2bn',dagnn.BatchNorm(),{'fc2'},{'fc2bn'},...
    {'fc2bn_w','fc2bn_b','fc2bn_m'});
%net.addLayer('fc2x',dagnn.ReLU(),{'fc2bn'},{'fc2bnx'});
%net.addLayer('dropout_diction',dagnn.DropOut('rate',0.5),{'fc2bn'},{'fc2bnd'});
% 32*256
convBlock = dagnn.Conv('size',[1 1 300 128],'hasBias',false,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('fc2_1_1',convBlock,{'fc2bn'},{'fc2_1_1'},{'fc2_1_1f'});
net.addLayer('fc2_1_1bn',dagnn.BatchNorm(),{'fc2_1_1'},{'fc2_1_1bn'},...
    {'fc2_1_1bn_w','fc2_1_1bn_b','fc2_1_1bn_m'});
net.addLayer('fc2_1_1x',dagnn.ReLU(),{'fc2_1_1bn'},{'fc2_1_1bnx'});

convBlock = dagnn.Conv('size',[1 2 128 128],'hasBias',false,'stride',[1,1],'pad',[0,0,1,0]);
net.addLayer('fc2_1_2',convBlock,{'fc2_1_1bnx'},{'fc2_1_2'},{'fc2_1_2f'});
net.addLayer('fc2_1_2bn',dagnn.BatchNorm(),{'fc2_1_2'},{'fc2_1_2bn'},...
    {'fc2_1_2bn_w','fc2_1_2bn_b','fc2_1_2bn_m'});
net.addLayer('fc2_1_2x',dagnn.ReLU(),{'fc2_1_2bn'},{'fc2_1_2bnx'});

convBlock = dagnn.Conv('size',[1 1 128 256],'hasBias',false,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('fc2_1_3',convBlock,{'fc2_1_2bnx'},{'fc2_1_3'},{'fc2_1_3f'});
net.addLayer('fc2_1_3bn',dagnn.BatchNorm(),{'fc2_1_3'},{'fc2_1_3bn'},...
    {'fc2_1_3bn_w','fc2_1_3bn_b','fc2_1_3bn_m'});

convBlock = dagnn.Conv('size',[1 1 300 256],'hasBias',false,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('fc2_1b',convBlock,{'fc2bn'},{'fc2_1b'},{'fc2_1bf'});
net.addLayer('fc2_1bbn',dagnn.BatchNorm(),{'fc2_1b'},{'fc2_1bbn'},...
    {'fc2_1bbn_w','fc2_1bbn_b','fc2_1bbn_m'});

net.addLayer('fc2_1sum',dagnn.Sum(),{'fc2_1_3bn','fc2_1bbn'},...
    {'fc2_1sum'});
net.addLayer('fc2_1x',dagnn.ReLU(),{'fc2_1sum'},{'fc2_1sumx'});

% 32*256
for i  = 2:3
    convBlock = dagnn.Conv('size',[1 1 256 64],'hasBias',false,'stride',[1,1],'pad',[0,0,0,0]);
    net.addLayer( sprintf('fc2_%d_1',i),convBlock,{sprintf('fc2_%dsumx',i-1)},{sprintf('fc2_%d_1',i)}, ...
        {sprintf('fc2_%d_1f',i)});
    net.addLayer(sprintf('fc2_%d_1bn_1',i),dagnn.BatchNorm(),{sprintf('fc2_%d_1',i)},{sprintf('fc2_%d_1bn',i)},...
        {sprintf('fc2_%d_1bn_w',i),sprintf('fc2_%d_1bn_b',i),sprintf('fc2_%d_1bn_m',i)});
    net.addLayer(sprintf('fc2_%d_1x',i),dagnn.ReLU(),{sprintf('fc2_%d_1bn',i)},{sprintf('fc2_%d_1bnx',i)});
    
    convBlock = dagnn.Conv('size',[1 2 64 64],'hasBias',false,'stride',[1,1],'pad',[0,0,1,0]);
    net.addLayer( sprintf('fc2_%d_2',i),convBlock,{sprintf('fc2_%d_1bnx',i)},{sprintf('fc2_%d_2',i)}, ...
        {sprintf('fc2_%d_2f',i)});
    net.addLayer(sprintf('fc2_%d_2bn',i),dagnn.BatchNorm(),{sprintf('fc2_%d_2',i)},{sprintf('fc2_%d_2bn',i)},...
        {sprintf('fc2_%d_2bn_w',i),sprintf('fc2_%d_2bn_b',i),sprintf('fc2_%d_2bn_m',i)});
    net.addLayer(sprintf('fc2_%d_2x',i),dagnn.ReLU(),{sprintf('fc2_%d_2bn',i)},{sprintf('fc2_%d_2bnx',i)});
    
    convBlock = dagnn.Conv('size',[1 1 64 256],'hasBias',false,'stride',[1,1],'pad',[0,0,0,0]);
    net.addLayer( sprintf('fc2_%d_3',i),convBlock,{sprintf('fc2_%d_2bnx',i)},{sprintf('fc2_%d_3',i)}, ...
        {sprintf('fc2_%d_3f',i)});
    net.addLayer(sprintf('fc2_%d_3bn',i),dagnn.BatchNorm(),{sprintf('fc2_%d_3',i)},{sprintf('fc2_%d_3bn',i)},...
        {sprintf('fc2_%d_3bn_w',i),sprintf('fc2_%d_3bn_b',i),sprintf('fc2_%d_3bn_m',i)});
    
    net.addLayer(sprintf('fc2_%dsum',i),dagnn.Sum(),{sprintf('fc2_%dsumx',i-1),sprintf('fc2_%d_3bn',i)},...
        {sprintf('fc2_%dsum',i)});
    net.addLayer(sprintf('fc2_%dx',i),dagnn.ReLU(),{sprintf('fc2_%dsum',i)},{sprintf('fc2_%dsumx',i)});
end

%32*256
convBlock = dagnn.Conv('size',[1 1 256 512],'hasBias',false,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('fc2_4a_1',convBlock,{'fc2_3sumx'},{'fc2_4a_1'},{'fc2_4a_1f'});
net.addLayer('fc2_4a_1bn',dagnn.BatchNorm(),{'fc2_4a_1'},{'fc2_4a_1bn'},...
    {'fc2_4a_1bn_w','fc2_4a_1bn_b','fc2_4a_1bn_m'});
net.addLayer('fc2_4a_1x',dagnn.ReLU(),{'fc2_4a_1bn'},{'fc2_4a_1bnx'});

convBlock = dagnn.Conv('size',[1 2 512 512],'hasBias',false,'stride',[2,2],'pad',[0,0,1,0]);
net.addLayer('fc2_4a_2',convBlock,{'fc2_4a_1bnx'},{'fc2_4a_2'},{'fc2_4a_2f'});
net.addLayer('fc2_4a_2bn',dagnn.BatchNorm(),{'fc2_4a_2'},{'fc2_4a_2bn'},...
    {'fc2_4a_2bn_w','fc2_4a_2bn_b','fc2_4a_2bn_m'});
net.addLayer('fc2_4a_2x',dagnn.ReLU(),{'fc2_4a_2bn'},{'fc2_4a_2bnx'});


convBlock = dagnn.Conv('size',[1 1 512 512],'hasBias',false,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('fc2_4a_3',convBlock,{'fc2_4a_2bnx'},{'fc2_4a_3'},{'fc2_4a_3f'});
net.addLayer('fc2_4a_3bn',dagnn.BatchNorm(),{'fc2_4a_3'},{'fc2_4a_3bn'},...
    {'fc2_4a_3bn_w','fc2_4a_3bn_b','fc2_4a_3bn_m'});

convBlock = dagnn.Conv('size',[1 1 256 512],'hasBias',false,'stride',[2,2],'pad',[0,0,0,0]);
net.addLayer('fc2_4b',convBlock,{'fc2_3sumx'},{'fc2_4b'},{'fc2_4bf'});
net.addLayer('fc2_4bbn',dagnn.BatchNorm(),{'fc2_4b'},{'fc2_4bbn'},...
    {'fc2_4bbn_w','fc2_4bbn_b','fc2_4bbn_m'});

%16*512
net.addLayer('fc2_4sum',dagnn.Sum(),{'fc2_4a_3bn','fc2_4bbn'},...
    {'fc2_4sum'});
net.addLayer('fc2_4x',dagnn.ReLU(),{'fc2_4sum'},{'fc3_1sumx'});

for i  = 2:4
    convBlock = dagnn.Conv('size',[1 1 512 128],'hasBias',false,'stride',[1,1],'pad',[0,0,0,0]);
    net.addLayer( sprintf('fc3_%d_1',i),convBlock,{sprintf('fc3_%dsumx',i-1)},{sprintf('fc3_%d_1',i)}, ...
        {sprintf('fc3_%d_1f',i)});
    net.addLayer(sprintf('fc3_%d_1bn_1',i),dagnn.BatchNorm(),{sprintf('fc3_%d_1',i)},{sprintf('fc3_%d_1bn',i)},...
        {sprintf('fc3_%d_1bn_w',i),sprintf('fc3_%d_1bn_b',i),sprintf('fc3_%d_1bn_m',i)});
    net.addLayer(sprintf('fc3_%d_1x',i),dagnn.ReLU(),{sprintf('fc3_%d_1bn',i)},{sprintf('fc3_%d_1bnx',i)});
    
    convBlock = dagnn.Conv('size',[1 2 128 128],'hasBias',false,'stride',[1,1],'pad',[0,0,1,0]);
    net.addLayer( sprintf('fc3_%d_2',i),convBlock,{sprintf('fc3_%d_1bnx',i)},{sprintf('fc3_%d_2',i)}, ...
        {sprintf('fc3_%d_2f',i)});
    net.addLayer(sprintf('fc3_%d_2bn',i),dagnn.BatchNorm(),{sprintf('fc3_%d_2',i)},{sprintf('fc3_%d_2bn',i)},...
        {sprintf('fc3_%d_2bn_w',i),sprintf('fc3_%d_2bn_b',i),sprintf('fc3_%d_2bn_m',i)});
    net.addLayer(sprintf('fc3_%d_2x',i),dagnn.ReLU(),{sprintf('fc3_%d_2bn',i)},{sprintf('fc3_%d_2bnx',i)});
    
    convBlock = dagnn.Conv('size',[1 1 128 512],'hasBias',false,'stride',[1,1],'pad',[0,0,0,0]);
    net.addLayer( sprintf('fc3_%d_3',i),convBlock,{sprintf('fc3_%d_2bnx',i)},{sprintf('fc3_%d_3',i)}, ...
        {sprintf('fc3_%d_3f',i)});
    net.addLayer(sprintf('fc3_%d_3bn',i),dagnn.BatchNorm(),{sprintf('fc3_%d_3',i)},{sprintf('fc3_%d_3bn',i)},...
        {sprintf('fc3_%d_3bn_w',i),sprintf('fc3_%d_3bn_b',i),sprintf('fc3_%d_3bn_m',i)});
    
    net.addLayer(sprintf('fc3_%dsum',i),dagnn.Sum(),{sprintf('fc3_%dsumx',i-1),sprintf('fc3_%d_3bn',i)},...
        {sprintf('fc3_%dsum',i)});
    net.addLayer(sprintf('fc3_%dx',i),dagnn.ReLU(),{sprintf('fc3_%dsum',i)},{sprintf('fc3_%dsumx',i)});
end

convBlock = dagnn.Conv('size',[1 1 512 1024],'hasBias',false,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('fc3_5a_1',convBlock,{'fc3_4sumx'},{'fc3_5a_1'},{'fc3_5a_1f'});
net.addLayer('fc3_5a_1bn',dagnn.BatchNorm(),{'fc3_5a_1'},{'fc3_5a_1bn'},...
    {'fc3_5a_1bn_w','fc3_5a_1bn_b','fc3_5a_1bn_m'});
net.addLayer('fc3_5a_1x',dagnn.ReLU(),{'fc3_5a_1bn'},{'fc3_5a_1bnx'});

convBlock = dagnn.Conv('size',[1 2 1024 1024],'hasBias',false,'stride',[2,2],'pad',[0,0,1,0]);
net.addLayer('fc3_5a_2',convBlock,{'fc3_5a_1bnx'},{'fc3_5a_2'},{'fc3_5a_2f'});
net.addLayer('fc3_5a_2bn',dagnn.BatchNorm(),{'fc3_5a_2'},{'fc3_5a_2bn'},...
    {'fc3_5a_2bn_w','fc3_5a_2bn_b','fc3_5a_2bn_m'});
net.addLayer('fc3_5a_2x',dagnn.ReLU(),{'fc3_5a_2bn'},{'fc3_5a_2bnx'});

convBlock = dagnn.Conv('size',[1 1 1024 1024],'hasBias',false,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('fc3_5a_3',convBlock,{'fc3_5a_2bnx'},{'fc3_5a_3'},{'fc3_5a_3f'});
net.addLayer('fc3_5a_3bn',dagnn.BatchNorm(),{'fc3_5a_3'},{'fc3_5a_3bn'},...
    {'fc3_5a_3bn_w','fc3_5a_3bn_b','fc3_5a_3bn_m'});

convBlock = dagnn.Conv('size',[1 1 512 1024],'hasBias',false,'stride',[2,2],'pad',[0,0,0,0]);
net.addLayer('fc3_5b',convBlock,{'fc3_4sumx'},{'fc3_5b'},{'fc3_5bf'});
net.addLayer('fc3_5bbn',dagnn.BatchNorm(),{'fc3_5b'},{'fc3_5bbn'},...
    {'fc3_5bbn_w','fc3_5bbn_b','fc3_5bbn_m'});

%8*1024
net.addLayer('fc3_5sum',dagnn.Sum(),{'fc3_5a_3bn','fc3_5bbn'},...
    {'fc3_5sum'});
net.addLayer('fc3_5x',dagnn.ReLU(),{'fc3_5sum'},{'fc4_1sumx'});

for i  = 2:6
    convBlock = dagnn.Conv('size',[1 1 1024 256],'hasBias',false,'stride',[1,1],'pad',[0,0,0,0]);
    net.addLayer( sprintf('fc4_%d_1',i),convBlock,{sprintf('fc4_%dsumx',i-1)},{sprintf('fc4_%d_1',i)}, ...
        {sprintf('fc4_%d_1f',i)});
    net.addLayer(sprintf('fc4_%d_1bn_1',i),dagnn.BatchNorm(),{sprintf('fc4_%d_1',i)},{sprintf('fc4_%d_1bn',i)},...
        {sprintf('fc4_%d_1bn_w',i),sprintf('fc4_%d_1bn_b',i),sprintf('fc4_%d_1bn_m',i)});
    net.addLayer(sprintf('fc4_%d_1x',i),dagnn.ReLU(),{sprintf('fc4_%d_1bn',i)},{sprintf('fc4_%d_1bnx',i)});
    
    convBlock = dagnn.Conv('size',[1 2 256 256],'hasBias',false,'stride',[1,1],'pad',[0,0,1,0]);
    net.addLayer( sprintf('fc4_%d_2',i),convBlock,{sprintf('fc4_%d_1bnx',i)},{sprintf('fc4_%d_2',i)}, ...
        {sprintf('fc4_%d_2f',i)});
    net.addLayer(sprintf('fc4_%d_2bn',i),dagnn.BatchNorm(),{sprintf('fc4_%d_2',i)},{sprintf('fc4_%d_2bn',i)},...
        {sprintf('fc4_%d_2bn_w',i),sprintf('fc4_%d_2bn_b',i),sprintf('fc4_%d_2bn_m',i)});
    net.addLayer(sprintf('fc4_%d_2x',i),dagnn.ReLU(),{sprintf('fc4_%d_2bn',i)},{sprintf('fc4_%d_2bnx',i)});
    
    convBlock = dagnn.Conv('size',[1 1 256 1024],'hasBias',false,'stride',[1,1],'pad',[0,0,0,0]);
    net.addLayer( sprintf('fc4_%d_3',i),convBlock,{sprintf('fc4_%d_2bnx',i)},{sprintf('fc4_%d_3',i)}, ...
        {sprintf('fc4_%d_3f',i)});
    net.addLayer(sprintf('fc4_%d_3bn',i),dagnn.BatchNorm(),{sprintf('fc4_%d_3',i)},{sprintf('fc4_%d_3bn',i)},...
        {sprintf('fc4_%d_3bn_w',i),sprintf('fc4_%d_3bn_b',i),sprintf('fc4_%d_3bn_m',i)});
    
    net.addLayer(sprintf('fc4_%dsum',i),dagnn.Sum(),{sprintf('fc4_%dsumx',i-1),sprintf('fc4_%d_3bn',i)},...
        {sprintf('fc4_%dsum',i)});
    net.addLayer(sprintf('fc4_%dx',i),dagnn.ReLU(),{sprintf('fc4_%dsum',i)},{sprintf('fc4_%dsumx',i)});
end

convBlock = dagnn.Conv('size',[1 1 1024 2048],'hasBias',false,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('fc4_7a_1',convBlock,{'fc4_6sumx'},{'fc4_7a_1'},{'fc4_7a_1f'});
net.addLayer('fc4_7a_1bn',dagnn.BatchNorm(),{'fc4_7a_1'},{'fc4_7a_1bn'},...
    {'fc4_7a_1bn_w','fc4_7a_1bn_b','fc4_7a_1bn_m'});
net.addLayer('fc4_7a_1x',dagnn.ReLU(),{'fc4_7a_1bn'},{'fc4_7a_1bnx'});

convBlock = dagnn.Conv('size',[1 2 2048 2048],'hasBias',false,'stride',[1,1],'pad',[0,0,1,0]);
net.addLayer('fc4_7a_2',convBlock,{'fc4_7a_1bnx'},{'fc4_7a_2'},{'fc4_7a_2f'});
net.addLayer('fc4_7a_2bn',dagnn.BatchNorm(),{'fc4_7a_2'},{'fc4_7a_2bn'},...
    {'fc4_7a_2bn_w','fc4_7a_2bn_b','fc4_7a_2bn_m'});
net.addLayer('fc4_7a_2x',dagnn.ReLU(),{'fc4_7a_2bn'},{'fc4_7a_2bnx'});

convBlock = dagnn.Conv('size',[1 1 2048 2048],'hasBias',false,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('fc4_7a_3',convBlock,{'fc4_7a_2bnx'},{'fc4_7a_3'},{'fc4_7a_3f'});
net.addLayer('fc4_7a_3bn',dagnn.BatchNorm(),{'fc4_7a_3'},{'fc4_7a_3bn'},...
    {'fc4_7a_3bn_w','fc4_7a_3bn_b','fc4_7a_3bn_m'});

convBlock = dagnn.Conv('size',[1 1 1024 2048],'hasBias',false,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('fc4_7b',convBlock,{'fc4_6sumx'},{'fc4_7b'},{'fc4_7bf'});
net.addLayer('fc4_7bbn',dagnn.BatchNorm(),{'fc4_7b'},{'fc4_7bbn'},...
    {'fc4_7bbn_w','fc4_7bbn_b','fc4_7bbn_m'});

%8*2048
net.addLayer('fc4_7sum',dagnn.Sum(),{'fc4_7a_3bn','fc4_7bbn'},...
    {'fc4_7sum'});
net.addLayer('fc4_7x',dagnn.ReLU(),{'fc4_7sum'},{'fc5_1sumx'});

poolBlock = dagnn.Pooling('poolSize',[1 8]);
net.addLayer('fc5_1',poolBlock,{'fc5_1sumx'},{'fc5_1bnx'});
%net.addLayer('fc5_1bn',dagnn.BatchNorm(),{'fc5_1'},{'fc5_1bn'},...
 %   {'fc5_1bn_w','fc5_1bn_b','fc5_1bn_m'});
%net.addLayer('fc5_1x',dagnn.ReLU(),{'fc5_1bn'},{'fc5_1bnx'});

fc5_2Block = dagnn.Conv('size',[1 1 2048 2048],'hasBias',false,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('fc5_2',fc5_2Block,{'fc5_1bnx'},{'fc5_2'},{'fc5_2f'});
net.addLayer('fc5_2bn',dagnn.BatchNorm(),{'fc5_2'},{'fc5_2bn'},...
    {'fc5_2bn_w','fc5_2bn_b','fc5_2bn_m'});
net.addLayer('fc5_2x',dagnn.ReLU(),{'fc5_2bn'},{'fc5_2bnx'});
net.addLayer('dropout2',dagnn.DropOut('rate',0.5),{'fc5_2bnx'},{'fc5_2bnxd'});

%----------------------add share layer----------------------
%1

fc_imgBlock = dagnn.Conv('size',[1 1 2048 113287],'hasBias',false,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('fc_img',fc_imgBlock,{'fc1_1bnxd'},{'prediction_img'},{'fcsharef'});
net.addLayer('softmaxloss_img',dagnn.Loss('loss','softmaxlog'),{'prediction_img','label_img'},'objective_img');
net.addLayer('top1err_img', dagnn.Loss('loss', 'classerror'), ...
    {'prediction_img','label_img'}, 'top1err_img') ;
net.addLayer('top5err_img', dagnn.Loss('loss', 'topkerror', ...
    'opts', {'topK',5}), ...
    {'prediction_img','label_img'}, 'top5err_img') ;
%2
fc_txtBlock = dagnn.Conv('size',[1 1 2048 113287],'hasBias',false,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('fc_txt',fc_txtBlock,{'fc5_2bnxd'},{'prediction_txt'},{'fcsharef'});
net.addLayer('softmaxloss_txt',dagnn.Loss('loss','softmaxlog'),{'prediction_txt','label_txt'},'objective_txt');
net.addLayer('top1err_txt', dagnn.Loss('loss', 'classerror'), ...
    {'prediction_txt','label_txt'}, 'top1err_txt') ;
net.addLayer('top5err_txt', dagnn.Loss('loss', 'topkerror', ...
    'opts', {'topK',5}), ...
    {'prediction_txt','label_txt'}, 'top5err_txt') ;
%}

%}
net.initParams();
%----------NOTICE--------------
first = net.getParamIndex('fc2f');

net.params(first).learningRate = 1e-3;  %w
net.params(first+1).learningRate = 1e-3;  %b

load('./dataset/MSCOCO-prepare/COCO_dictionary.mat');
net.params(first).value = reshape(single(subset.features'),1,1,29972,300);

end

