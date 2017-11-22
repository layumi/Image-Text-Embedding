function net = vgg_2stream_small4x_2a()
% this code is written by zhedong zheng
% matconvnet model
net = dagnn.DagNN();
reluBlock = dagnn.ReLU('leak',0.1);

%------------pic process
%conv (224-3+2)/2+1 = 112   (112-3)/2+1 = 55   (55-3)/2+1 = 27
%conv1_1Block = dagnn.Conv('size',[3 3 10 96],'hasBias',true,'stride',[2,2],'pad',[1,1,1,1]);
conv1_2Block = dagnn.Conv('size',[3 3 10 96],'hasBias',true,'stride',[2,2],'pad',[0,0,0,0]);
conv1_3Block = dagnn.Conv('size',[3 3 96 96],'hasBias',true,'stride',[1,1],'pad',[1,1,1,1]);
for i=1:2
    %net.addLayer(sprintf('conv1_1_%d',i),conv1_1Block,{sprintf('input%d',i)},{sprintf('conv1_1_%d',i)},{sprintf('c1_1f_%d',i),sprintf('c1_1b_%d',i)});
    %net.addLayer(sprintf('bn1_1_%d',i),dagnn.BatchNorm(),{sprintf('conv1_1_%d',i)},{sprintf('conv1_1_%dbn',i)},{sprintf('bn1_1f_%d',i),sprintf('bn1_1b_%d',i),sprintf('bn1_1c_%d',i)});
    %net.addLayer(sprintf('relu1_1_%d',i),reluBlock,{sprintf('conv1_1_%dbn',i)},{sprintf('conv1_1_%dx',i)},{});
    %pool 0
    %net.addLayer(sprintf('pool0_%d',i),dagnn.Pooling('poolSize',[3,3],'stride',[2,2],'pad',[0,0,0,0]),...
     %   {sprintf('conv1_1_%dx',i)},{sprintf('conv1_1_%dp',i)});
    net.addLayer(sprintf('conv1_2_%d',i),conv1_2Block,{sprintf('input%d',i)},{sprintf('conv1_2_%d',i)},{sprintf('c1_2f_%d',i),sprintf('c1_2b_%d',i)});
    net.addLayer(sprintf('bn1_2_%d',i),dagnn.BatchNorm(),{sprintf('conv1_2_%d',i)},{sprintf('conv1_2_%dbn',i)},{sprintf('bn1_2f_%d',i),sprintf('bn1_2b_%d',i),sprintf('bn1_2c_%d',i)});
    net.addLayer(sprintf('relu1_2_%d',i),reluBlock,{sprintf('conv1_2_%dbn',i)},{sprintf('conv1_2_%dx',i)},{});
    net.addLayer(sprintf('conv1_3_%d',i),conv1_3Block,{sprintf('conv1_2_%dx',i)},{sprintf('conv1_3_%d',i)},{sprintf('c1_3f_%d',i),sprintf('c1_3b_%d',i)});
    net.addLayer(sprintf('bn1_3_%d',i),dagnn.BatchNorm(),{sprintf('conv1_3_%d',i)},{sprintf('conv1_3_%dbn',i)},{sprintf('bn1_3f_%d',i),sprintf('bn1_3b_%d',i),sprintf('bn1_3c_%d',i)});
    net.addLayer(sprintf('relu1_3_%d',i),reluBlock,{sprintf('conv1_3_%dbn',i)},{sprintf('conv1_3_%dx',i)},{});
end
%concat
cat1Block = dagnn.Concat();
net.addLayer('concat1',cat1Block,{'conv1_3_1x','conv1_3_2x'},{'concat1'},{});

conv2_1Block = dagnn.Conv('size',[3 3 192 288],'hasBias',true,'stride',[1,1],'pad',[1,1,1,1]);
net.addLayer('conv2_1',conv2_1Block,{'concat1'},{'conv2_1'},{'c2_1f','c2_1b'});
net.addLayer('bn2_1',dagnn.BatchNorm(),{'conv2_1'},{'conv2_1bn'},{'bn2_1f','bn2_1b','bn2_1c'});
net.addLayer('relu2_1',reluBlock,{'conv2_1bn'},{'conv2x'},{});

% input 27 * 27 * 288 output 14*14*768
%----------------inception figure 5
%1 1*1+3*3
conv3_1Block = dagnn.Conv('size',[1 1 288 192],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('conv3_1',conv3_1Block,{'conv2x'},{'conv3_1'},{'c3_1f','c3_1b'});
net.addLayer('bn3_1',dagnn.BatchNorm(),{'conv3_1'},{'conv3_1bn'},{'bn3_1f','bn3_1b','bn3_1c'});
net.addLayer('relu3_1',reluBlock,{'conv3_1bn'},{'conv3_1x'},{});
conv3_2Block = dagnn.Conv('size',[3 3 192 192],'hasBias',true,'stride',[2,2],'pad',[1,1,1,1]);
net.addLayer('conv3_2',conv3_2Block,{'conv3_1x'},{'conv3_2'},{'c3_2f','c3_2b'});
net.addLayer('bn3_2',dagnn.BatchNorm(),{'conv3_2'},{'conv3_2bn'},{'bn3_2f','bn3_2b','bn3_2c'});
net.addLayer('relu3_2',reluBlock,{'conv3_2bn'},{'conv3_2x'},{});

%2 1*1
conv3_3Block = dagnn.Conv('size',[1 1 288 192],'hasBias',true,'stride',[2,2],'pad',[0,0,0,0]);
net.addLayer('conv3_3',conv3_3Block,{'conv2x'},{'conv3_3'},{'c3_3f','c3_3b'});
net.addLayer('bn3_3',dagnn.BatchNorm(),{'conv3_3'},{'conv3_3bn'},{'bn3_3f','bn3_3b','bn3_3c'});
net.addLayer('relu3_3',reluBlock,{'conv3_3bn'},{'conv3_3x'},{});

%3 pool 1*1
pool3_4Block = dagnn.Pooling('poolSize',[3,3],'stride',[2,2],'pad',[1,1,1,1]);
net.addLayer('pool3_4',pool3_4Block,{'conv2x'},{'pool3_4'});
conv3_5Block = dagnn.Conv('size',[1 1 288 192],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('conv3_5',conv3_5Block,{'pool3_4'},{'conv3_5'},{'c3_5f','c3_5b'});
net.addLayer('bn3_5',dagnn.BatchNorm(),{'conv3_5'},{'conv3_5bn'},{'bn3_5f','bn3_5b','bn3_5c'});
net.addLayer('relu3_5',reluBlock,{'conv3_5bn'},{'conv3_5x'},{});

%4 1*1+3*3+3*3
conv3_6Block = dagnn.Conv('size',[1 1 288 192],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('conv3_6',conv3_6Block,{'conv2x'},{'conv3_6'},{'c3_6f','c3_6b'});
net.addLayer('bn3_6',dagnn.BatchNorm(),{'conv3_6'},{'conv3_6bn'},{'bn3_6f','bn3_6b','bn3_6c'});
net.addLayer('relu3_6',reluBlock,{'conv3_6bn'},{'conv3_6x'},{});
conv3_7Block = dagnn.Conv('size',[3 3 192 192],'hasBias',true,'stride',[1,1],'pad',[1,1,1,1]);
net.addLayer('conv3_7',conv3_7Block,{'conv3_6x'},{'conv3_7'},{'c3_7f','c3_7b'});
net.addLayer('bn3_7',dagnn.BatchNorm(),{'conv3_7'},{'conv3_7bn'},{'bn3_7f','bn3_7b','bn3_7c'});
net.addLayer('relu3_7',reluBlock,{'conv3_7bn'},{'conv3_7x'},{});
conv3_8Block = dagnn.Conv('size',[3 3 192 192],'hasBias',true,'stride',[2,2],'pad',[1,1,1,1]);
net.addLayer('conv3_8',conv3_8Block,{'conv3_7x'},{'conv3_8'},{'c3_8f','c3_8b'});
net.addLayer('bn3_8',dagnn.BatchNorm(),{'conv3_8'},{'conv3_8bn'},{'bn3_8f','bn3_8b','bn3_8c'});
net.addLayer('relu3_8',reluBlock,{'conv3_8bn'},{'conv3_8x'},{});
%concat
cat3Block = dagnn.Concat();
net.addLayer('concat3',cat3Block,{'conv3_2x','conv3_3x','conv3_5x','conv3_8x'},{'concat3'},{});

% input 14 * 14 * 768 output 7 * 7 * 1280
%----------------inception figure 6  
%1 1*1+1*5+5*1
conv4_1Block = dagnn.Conv('size',[1 1 768 320],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('conv4_1',conv4_1Block,{'concat3'},{'conv4_1'},{'c4_1f','c4_1b'});
net.addLayer('bn4_1',dagnn.BatchNorm(),{'conv4_1'},{'conv4_1bn'},{'bn4_1f','bn4_1b','bn4_1c'});
net.addLayer('relu4_1',reluBlock,{'conv4_1bn'},{'conv4_1x'},{});
conv4_2Block = dagnn.Conv('size',[1 5 320 320],'hasBias',true,'stride',[1,2],'pad',[0,0,2,2]);
net.addLayer('conv4_2',conv4_2Block,{'conv4_1x'},{'conv4_2'},{'c4_2f','c4_2b'});
net.addLayer('bn4_2',dagnn.BatchNorm(),{'conv4_2'},{'conv4_2bn'},{'bn4_2f','bn4_2b','bn4_2c'});
net.addLayer('relu4_2',reluBlock,{'conv4_2bn'},{'conv4_2x'},{});
conv4_3Block = dagnn.Conv('size',[5 1 320 320],'hasBias',true,'stride',[2,1],'pad',[2,2,0,0]);
net.addLayer('conv4_3',conv4_3Block,{'conv4_2x'},{'conv4_3'},{'c4_3f','c4_3b'});
net.addLayer('bn4_3',dagnn.BatchNorm(),{'conv4_3'},{'conv4_3bn'},{'bn4_3f','bn4_3b','bn4_3c'});
net.addLayer('relu4_3',reluBlock,{'conv4_3bn'},{'conv4_3x'},{});

%2 1*1
conv4_4Block = dagnn.Conv('size',[1 1 768 320],'hasBias',true,'stride',[2,2],'pad',[0,0,0,0]);
net.addLayer('conv4_4',conv4_4Block,{'concat3'},{'conv4_4'},{'c4_4f','c4_4b'});
net.addLayer('bn4_4',dagnn.BatchNorm(),{'conv4_4'},{'conv4_4bn'},{'bn4_4f','bn4_4b','bn4_4c'});
net.addLayer('relu4_4',reluBlock,{'conv4_4bn'},{'conv4_4x'},{});

%3 pool 1*1
pool4_5Block = dagnn.Pooling('poolSize',[3,3],'stride',[2,2],'pad',[1,1,1,1]);
net.addLayer('pool4_5',pool4_5Block,{'concat3'},{'pool4_5'});
conv4_6Block = dagnn.Conv('size',[1 1 768 320],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('conv4_6',conv4_6Block,{'pool4_5'},{'conv4_6'},{'c4_6f','c4_6b'});
net.addLayer('bn4_6',dagnn.BatchNorm(),{'conv4_6'},{'conv4_6bn'},{'bn4_6f','bn4_6b','bn4_6c'});
net.addLayer('relu4_6',reluBlock,{'conv4_6bn'},{'conv4_6x'},{});

%4 1*1+1*5+5*1+1*5+5*1
conv4_7Block = dagnn.Conv('size',[1 1 768 320],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('conv4_7',conv4_7Block,{'concat3'},{'conv4_7'},{'c4_7f','c4_7b'});
net.addLayer('bn4_7',dagnn.BatchNorm(),{'conv4_7'},{'conv4_7bn'},{'bn4_7f','bn4_7b','bn4_7c'});
net.addLayer('relu4_7',reluBlock,{'conv4_7bn'},{'conv4_7x'},{});
conv4_8Block = dagnn.Conv('size',[1 5 320 320],'hasBias',true,'stride',[1,1],'pad',[0,0,2,2]);
net.addLayer('conv4_8',conv4_8Block,{'conv4_7x'},{'conv4_8'},{'c4_8f','c4_8b'});
net.addLayer('bn4_8',dagnn.BatchNorm(),{'conv4_8'},{'conv4_8bn'},{'bn4_8f','bn4_8b','bn4_8c'});
net.addLayer('relu4_8',reluBlock,{'conv4_8bn'},{'conv4_8x'},{});
conv4_9Block = dagnn.Conv('size',[5 1 320 320],'hasBias',true,'stride',[1,1],'pad',[2,2,0,0]);
net.addLayer('conv4_9',conv4_9Block,{'conv4_8x'},{'conv4_9'},{'c4_9f','c4_9b'});
net.addLayer('bn4_9',dagnn.BatchNorm(),{'conv4_9'},{'conv4_9bn'},{'bn4_9f','bn4_9b','bn4_9c'});
net.addLayer('relu4_9',reluBlock,{'conv4_9bn'},{'conv4_9x'},{});
conv4_10Block = dagnn.Conv('size',[1 5 320 320],'hasBias',true,'stride',[1,2],'pad',[0,0,2,2]);
net.addLayer('conv4_10',conv4_10Block,{'conv4_9x'},{'conv4_10'},{'c4_10f','c4_10b'});
net.addLayer('bn4_10',dagnn.BatchNorm(),{'conv4_10'},{'conv4_10bn'},{'bn4_10f','bn4_10b','bn4_10c'});
net.addLayer('relu4_10',reluBlock,{'conv4_10bn'},{'conv4_10x'},{});
conv4_11Block = dagnn.Conv('size',[5 1 320 320],'hasBias',true,'stride',[2,1],'pad',[2,2,0,0]);
net.addLayer('conv4_11',conv4_11Block,{'conv4_10x'},{'conv4_11'},{'c4_11f','c4_11b'});
net.addLayer('bn4_11',dagnn.BatchNorm(),{'conv4_11'},{'conv4_11bn'},{'bn4_11f','bn4_11b','bn4_11c'});
net.addLayer('relu4_11',reluBlock,{'conv4_11bn'},{'conv4_11x'},{});

cat4Block = dagnn.Concat();
net.addLayer('concat1_1',cat4Block,{'conv4_3x','conv4_4x','conv4_6x','conv4_11x'},{'concat4'},{});

%------------inception figure 7
%input 7*7*1280 output 7*7*2048

%1 1*1+(1*3 3*1)
conv5_1Block = dagnn.Conv('size',[1 1 1280 512],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('conv5_1',conv5_1Block,{'concat4'},{'conv5_1'},{'c5_1f','c5_1b'});
net.addLayer('bn5_1',dagnn.BatchNorm(),{'conv5_1'},{'conv5_1bn'},{'bn5_1f','bn5_1b','bn5_1c'});
net.addLayer('relu5_1',reluBlock,{'conv5_1bn'},{'conv5_1x'},{});
conv5_2Block = dagnn.Conv('size',[1 3 512 256],'hasBias',true,'stride',[1,1],'pad',[0,0,1,1]);
net.addLayer('conv5_2',conv5_2Block,{'conv5_1x'},{'conv5_2'},{'c5_2f','c5_2b'});
net.addLayer('bn5_2',dagnn.BatchNorm(),{'conv5_2'},{'conv5_2bn'},{'bn5_2f','bn5_2b','bn5_2c'});
net.addLayer('relu5_2',reluBlock,{'conv5_2bn'},{'conv5_2x'},{});
conv5_3Block = dagnn.Conv('size',[3 1 512 256],'hasBias',true,'stride',[1,1],'pad',[1,1,0,0]);
net.addLayer('conv5_3',conv5_3Block,{'conv5_1x'},{'conv5_3'},{'c5_3f','c5_3b'});
net.addLayer('bn5_3',dagnn.BatchNorm(),{'conv5_3'},{'conv5_3bn'},{'bn5_3f','bn5_3b','bn5_3c'});
net.addLayer('relu5_3',reluBlock,{'conv5_3bn'},{'conv5_3x'},{});
cat5_1Block = dagnn.Concat();
net.addLayer('concat5_1',cat5_1Block,{'conv5_2x','conv5_3x'},{'concat5_1'},{});

%2 1*1
conv5_4Block = dagnn.Conv('size',[1 1 1280 512],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('conv5_4',conv5_4Block,{'concat4'},{'conv5_4'},{'c5_4f','c5_4b'});
net.addLayer('bn5_4',dagnn.BatchNorm(),{'conv5_4'},{'conv5_4bn'},{'bn5_4f','bn5_4b','bn5_4c'});
net.addLayer('relu5_4',reluBlock,{'conv5_4bn'},{'conv5_4x'},{});

%3 pool 1*1
pool5_5Block = dagnn.Pooling('poolSize',[3,3],'stride',[1,1],'pad',[1,1,1,1]);
net.addLayer('pool5_5',pool5_5Block,{'concat4'},{'pool5_5'});
conv5_6Block = dagnn.Conv('size',[1 1 1280 512],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('conv5_6',conv5_6Block,{'pool5_5'},{'conv5_6'},{'c5_6f','c5_6b'});
net.addLayer('bn5_6',dagnn.BatchNorm(),{'conv5_6'},{'conv5_6bn'},{'bn5_6f','bn5_6b','bn5_6c'});
net.addLayer('relu5_6',reluBlock,{'conv5_6bn'},{'conv5_6x'},{});

%4 1*1+3*3+(1*3 3*1)
conv5_7Block = dagnn.Conv('size',[1 1 1280 512],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('conv5_7',conv5_7Block,{'concat4'},{'conv5_7'},{'c5_7f','c5_7b'});
net.addLayer('bn5_7',dagnn.BatchNorm(),{'conv5_7'},{'conv5_7bn'},{'bn5_7f','bn5_7b','bn5_7c'});
net.addLayer('relu5_7',reluBlock,{'conv5_7bn'},{'conv5_7x'},{});
conv5_8Block = dagnn.Conv('size',[3 3 512 512],'hasBias',true,'stride',[1,1],'pad',[1,1,1,1]);
net.addLayer('conv5_8',conv5_8Block,{'conv5_7x'},{'conv5_8'},{'c5_8f','c5_8b'});
net.addLayer('bn5_8',dagnn.BatchNorm(),{'conv5_8'},{'conv5_8bn'},{'bn5_8f','bn5_8b','bn5_8c'});
net.addLayer('relu5_8',reluBlock,{'conv5_8bn'},{'conv5_8x'},{});
conv5_9_1Block = dagnn.Conv('size',[1 3 512 256],'hasBias',true,'stride',[1,1],'pad',[0,0,1,1]);
net.addLayer('conv5_9_1',conv5_9_1Block,{'conv5_8x'},{'conv5_9_1'},{'c5_9_1f','c5_9_1b'});
net.addLayer('bn5_9_1',dagnn.BatchNorm(),{'conv5_9_1'},{'conv5_9_1bn'},{'bn5_9_1f','bn5_9_1b','bn5_9_1c'});
net.addLayer('relu5_9_1',reluBlock,{'conv5_9_1bn'},{'conv5_9_1x'},{});
conv5_9_2Block = dagnn.Conv('size',[3 1 512 256],'hasBias',true,'stride',[1,1],'pad',[1,1,0,0]);
net.addLayer('conv5_9_2',conv5_9_2Block,{'conv5_8x'},{'conv5_9_2'},{'c5_9_2f','c5_9_2b'});
net.addLayer('bn5_9_2',dagnn.BatchNorm(),{'conv5_9_2'},{'conv5_9_2bn'},{'bn5_9_2f','bn5_9_2b','bn5_9_2c'});
net.addLayer('relu5_9_2',reluBlock,{'conv5_9_2bn'},{'conv5_9_2x'},{});
cat5_2Block = dagnn.Concat();
net.addLayer('concat5_2',cat5_2Block,{'conv5_9_1x','conv5_9_2x'},{'concat5_2'},{});

%concat  (13-3+2)/2+1= 7     *2048
cat5Block = dagnn.Concat();
net.addLayer('concat5',cat5Block,{'concat5_1','conv5_4x','conv5_6x','concat5_2'},{'concat5'},{});

%----------decide
pool6Block = dagnn.Pooling('poolSize',[7,7],'stride',[1,1],'pad',[0,0,0,0],'method','avg');
net.addLayer('pool6',pool6Block,{'concat5'},{'pool6'});
conv6_1Block = dagnn.Conv('size',[1 1 2048 1024],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('conv6_1',conv6_1Block,{'pool6'},{'conv6_1'},{'c6_1f','c6_1b'});
net.addLayer('bn6_1',dagnn.BatchNorm(),{'conv6_1'},{'conv6_1bn'},{'bn6_1f','bn6_1b','bn6_1c'});
net.addLayer('relu6_1',reluBlock,{'conv6_1bn'},{'conv6_1x'},{});

dropout4Block = dagnn.DropOut('rate',0.9); 
net.addLayer('dropout4',dropout4Block,{'conv6_1x'},{'conv12d'},{});

conv7Block = dagnn.Conv('size',[1,1,1024,101],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('conv7',conv7Block,{'conv12d'},{'prediction'},{'c7f','c7b'});

lossBlock = dagnn.Loss('loss', 'softmaxlog_ls');%label_smooth
net.addLayer('softmaxloss',lossBlock,{'prediction','label'},'objective');

net.addLayer('top1err', dagnn.Loss('loss', 'classerror'), ...
    {'prediction','label'}, 'top1err') ;
net.addLayer('top5err', dagnn.Loss('loss', 'topkerror', ...
    'opts', {'topK',5}), ...
    {'prediction','label'}, 'top5err') ;

%-------------- auxiliary
%pool 5x5 stride3
poolaBlock = dagnn.Pooling('poolSize',[5,5],'stride',[3,3],'pad',[0,0,0,0],'method','avg');
net.addLayer('poola',poolaBlock,{'concat3'},{'poola'});
% 1X1 + 4x4 
conv6_1aBlock = dagnn.Conv('size',[1 1 768 768],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('conv6_1a',conv6_1aBlock,{'poola'},{'conv6_1a'},{'c6_1af','c6_1ab'});
net.addLayer('bn6_1a',dagnn.BatchNorm(),{'conv6_1a'},{'conv6_1abn'},{'bn6_1af','bn6_1ab','bn6_1ac'});
net.addLayer('relu6_1a',reluBlock,{'conv6_1abn'},{'conv6_1ax'},{});
conv6_2aBlock = dagnn.Conv('size',[4 4 768 768],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('conv6_2a',conv6_2aBlock,{'conv6_1ax'},{'conv6_2a'},{'c6_2af','c6_2ab'});
net.addLayer('bn6_2a',dagnn.BatchNorm(),{'conv6_2a'},{'conv6_2abn'},{'bn6_2af','bn6_2ab','bn6_2ac'});
net.addLayer('relu6_2a',reluBlock,{'conv6_2abn'},{'conv6_2ax'},{});
dropoutaBlock = dagnn.DropOut('rate',0.9); 
net.addLayer('dropouta',dropoutaBlock,{'conv6_2ax'},{'conv12ad'},{});
%add bnorm
conv7aBlock = dagnn.Conv('size',[1,1,768,101],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('conv7a',conv7aBlock,{'conv12ad'},{'predictiona'},{'ca7f','ca7b'});
net.addLayer('bn7a',dagnn.BatchNorm(),{'predictiona'},{'predictiona_bn'},{'bna_f','bna_b','bna_c'});
lossBlock = dagnn.Loss('loss', 'softmaxlog_ls');%label_smooth
net.addLayer('softmaxloss_a',lossBlock,{'predictiona_bn','label'},'objective_a');
net.addLayer('top5err_a', dagnn.Loss('loss', 'topkerror', ...
    'opts', {'topK',5}), ...
    {'predictiona_bn','label'}, 'top5err_a') ;

net.initParams();
net.conserveMemory=true;
net.meta.inputSize = [224 224 2 10] ;
net.meta.trainOpts.learningRate = [0.01*ones(1,300) 0.001*ones(1,120)] ; %nouse
net.meta.trainOpts.weightDecay = 0.0005 ;
net.meta.trainOpts.batchSize = 256 ;
net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate) ;