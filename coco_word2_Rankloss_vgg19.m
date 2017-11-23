function net = resnet52_new_hope_RankLoss()

%----------------------load pretrained model----------------------
netStruct = load('./data/vgg19_coco_batch32_pool_shift_both_drop0.5/net-epoch-70.mat') ;
net = dagnn.DagNN.loadobj(netStruct.net) ;

for i = 1:36 %img cnn
    if(mod(i,2)==0)
        net.params(i).learningRate= 0.02;
    else net.params(i).learningRate= 0.001;
    end
    net.params(i).weightDecay=1;
end
net.params(1).learningRate = 0.0001;                                                                                                  

net.addLayer('lrn1',dagnn.LRN('param',[4096,1e-8,1,0.5]),{'fc1_1bnx'},{'fc1bn_n'},{});
net.addLayer('lrn2',dagnn.LRN('param',[4096,1e-8,1,0.5]),{'fc5_2bnx'},{'fc2bn_n'},{});

%--for get harder sample
%net.addLayer('Multiple',dagnn.Multiple(),{'fc1bn_n','fc2bn_n'},{'Score'},{});

lossBlock = dagnn.RankLoss('rate',1);
net.addLayer('RankLoss',lossBlock,{'fc1bn_n','fc2bn_n'},'objective_f');


%net.conserveMemory = false;
%net.eval({'data',single(rand(224,224,3)),'data2',single(rand(1,1,20074))});
end

