function net = flownet_nocrop()
net = dagnn.DagNN();
reluBlock = dagnn.ReLU('leak',0.1);

conv1Block = dagnn.Conv('size',[7 7 6 64],'hasBias',true,'stride',[2,2],'pad',[3,3,3,3]);
net.addLayer('conv1',conv1Block,{'input'},{'conv1'},{'c1f','c1b'});
net.addLayer('relu1',reluBlock,{'conv1'},{'conv1x'},{});

conv2Block = dagnn.Conv('size',[5 5 64 128],'hasBias',true,'stride',[2,2],'pad',[2,2,2,2]);
net.addLayer('conv2',conv2Block,{'conv1x'},{'conv2'},{'c2f','c2b'});
net.addLayer('relu2',reluBlock,{'conv2'},{'conv2x'},{});

conv3Block = dagnn.Conv('size',[5 5 128 256],'hasBias',true,'stride',[2,2],'pad',[2,2,2,2]);
net.addLayer('conv3',conv3Block,{'conv2x'},{'conv3'},{'c3f','c3b'});
net.addLayer('relu3',reluBlock,{'conv3'},{'conv3x'},{});
conv3_1Block = dagnn.Conv('size',[3 3 256 256],'hasBias',true,'stride',[1,1],'pad',[1,1,1,1]);
net.addLayer('conv3_1',conv3_1Block,{'conv3x'},{'conv3_1'},{'c3_1f','c3_1b'});
net.addLayer('relu4',reluBlock,{'conv3_1'},{'conv3_1x'},{});

conv4Block = dagnn.Conv('size',[3 3 256 512],'hasBias',true,'stride',[2,2],'pad',[1,1,1,1]);
net.addLayer('conv4',conv4Block,{'conv3_1x'},{'conv4'},{'c4f','c4b'});
net.addLayer('relu5',reluBlock,{'conv4'},{'conv4x'},{});
conv4_1Block = dagnn.Conv('size',[3 3 512 512],'hasBias',true,'stride',[1,1],'pad',[1,1,1,1]);
net.addLayer('conv4_1',conv4_1Block,{'conv4x'},{'conv4_1'},{'c4_1f','c4_1b'});
net.addLayer('relu6',reluBlock,{'conv4_1'},{'conv4_1x'},{});

conv5Block = dagnn.Conv('size',[3 3 512 512],'hasBias',true,'stride',[2,2],'pad',[1,1,1,1]);
net.addLayer('conv5',conv5Block,{'conv4_1x'},{'conv5'},{'c5f','c5b'});
net.addLayer('relu7',reluBlock,{'conv5'},{'conv5x'},{});
conv5_1Block = dagnn.Conv('size',[3 3 512 512],'hasBias',true,'stride',[1,1],'pad',[1,1,1,1]);
net.addLayer('conv5_1',conv5_1Block,{'conv5x'},{'conv5_1'},{'c5_1f','c5_1b'});
net.addLayer('relu8',reluBlock,{'conv5_1'},{'conv5_1x'},{});

conv6Block = dagnn.Conv('size',[3 3 512 1024],'hasBias',true,'stride',[2,2],'pad',[1,1,1,1]);
net.addLayer('conv6',conv6Block,{'conv5_1x'},{'conv6'},{'c6f','c6b'});
net.addLayer('relu9',reluBlock,{'conv6'},{'conv6x'},{});
conv6_1Block = dagnn.Conv('size',[3 3 1024 1024],'hasBias',true,'stride',[1,1],'pad',[1,1,1,1]);
net.addLayer('conv6_1',conv6_1Block,{'conv6x'},{'conv6_1'},{'c6_1f','c6_1b'});
net.addLayer('relu10',reluBlock,{'conv6_1'},{'conv6_1x'},{});

%------------------deconv part------------------
%result1
Conv1Block = dagnn.Conv('size',[3 3 1024 2],'hasBias',true,'stride',[1,1],'pad',[1,1,1,1]);
net.addLayer('Conv1',Conv1Block,{'conv6_1x'},{'predict_flow6'},{'Conv1f','Conv1b'});

%deconv1
deconv1Block = dagnn.ConvTranspose('size',[4 4 512 1024],'hasBias',true,'upsample',[2,2],'crop',[1,1,1,1]);
net.addLayer('deconv5',deconv1Block,{'conv6_1x'},{'deconv5'},{'dc5f','dc5b'});
net.addLayer('relu11',reluBlock,{'deconv5'},{'deconv5x'},{});
deconv1_1Block = dagnn.ConvTranspose('size',[4 4 2 2],'hasBias',true,'upsample',[2,2],'crop',[1,1,1,1]);
net.addLayer('upsampled_flow6_to_5',deconv1_1Block,{'predict_flow6'},{'upsampled_flow6_to_5'},{'result1dcf','result1dcb'});

cat2Block = dagnn.Concat();
net.addLayer('concat2',cat2Block,{'conv5_1x','deconv5x','upsampled_flow6_to_5'},{'concat5'});

%result2
Conv2Block = dagnn.Conv('size',[3 3 1026 2],'hasBias',true,'stride',[1,1],'pad',[1,1,1,1]);
net.addLayer('Conv2',Conv2Block,{'concat5'},{'predict_flow5'},{'Conv2f','Conv2b'});
%deconv2
deconv2Block = dagnn.ConvTranspose('size',[4 4 256 1026],'hasBias',true,'upsample',[2,2],'crop',[1,1,1,1]);
net.addLayer('deconv4',deconv2Block,{'concat5'},{'deconv4'},{'dc4f','dc4b'});
net.addLayer('relu12',reluBlock,{'deconv4'},{'deconv4x'},{});
deconv2_1Block = dagnn.ConvTranspose('size',[4 4 2 2],'hasBias',true,'upsample',[2,2],'crop',[1,1,1,1]);
net.addLayer('upsampled_flow5_to_4',deconv2_1Block,{'predict_flow5'},{'upsampled_flow5_to_4'},{'result2dcf','result2dcb'});

cat3Block = dagnn.Concat();
net.addLayer('concat3',cat3Block,{'conv4_1x','deconv4x','upsampled_flow5_to_4'},{'concat4'});

%result3
Conv3Block = dagnn.Conv('size',[3 3 770 2],'hasBias',true,'stride',[1,1],'pad',[1,1,1,1]);
net.addLayer('Conv3',Conv3Block,{'concat4'},{'predict_flow4'},{'Conv3f','Conv3b'});

%deconv3
deconv3Block = dagnn.ConvTranspose('size',[4 4 128 770],'hasBias',true,'upsample',[2,2],'crop',[1,1,1,1]);
net.addLayer('deconv3',deconv3Block,{'concat4'},{'deconv3'},{'dc3f','dc3b'});
net.addLayer('relu13',reluBlock,{'deconv3'},{'deconv3x'},{});
deconv3_1Block = dagnn.ConvTranspose('size',[4 4 2 2],'hasBias',true,'upsample',[2,2],'crop',[1,1,1,1]);
net.addLayer('upsampled_flow4_to_3',deconv3_1Block,{'predict_flow4'},{'upsampled_flow4_to_3'},{'result3dcf','result3dcb'});

cat4Block = dagnn.Concat();
net.addLayer('concat4',cat4Block,{'conv3_1x','deconv3x','upsampled_flow4_to_3'},{'concat3'});

%result4
Conv4Block = dagnn.Conv('size',[3 3 386 2],'hasBias',true,'stride',[1,1],'pad',[1,1,1,1]);
net.addLayer('Conv4',Conv4Block,{'concat3'},{'predict_flow3'},{'Conv4f','Conv4b'});
%deconv4
deconv4Block = dagnn.ConvTranspose('size',[4 4 64 386],'hasBias',true,'upsample',[2,2],'crop',[1,1,1,1]);
net.addLayer('deconv2',deconv4Block,{'concat3'},{'deconv2'},{'dc2f','dc2b'});
net.addLayer('relu14',reluBlock,{'deconv2'},{'deconv2x'},{});
deconv4_1Block = dagnn.ConvTranspose('size',[4 4 2 2],'hasBias',true,'upsample',[2,2],'crop',[1,1,1,1]);
net.addLayer('upsampled_flow3_to_2',deconv4_1Block,{'predict_flow3'},{'upsampled_flow3_to_2'},{'result4dcf','result4dcb'});

cat5Block = dagnn.Concat();
net.addLayer('concat5',cat5Block,{'conv2x','deconv2x','upsampled_flow3_to_2'},{'concat2'});

%result5
Conv5Block = dagnn.Conv('size',[3 3 194 2],'hasBias',true,'stride',[1,1],'pad',[1,1,1,1]);
net.addLayer('Conv5',Conv5Block,{'concat2'},{'predict_flow2'},{'Conv5f','Conv5b'});

%add loss
net.addLayer('loss5',EPELoss(),{'predict_flow5','label5'},'objective5');
net.addLayer('loss4',EPELoss(),{'predict_flow4','label4'},'objective4');
net.addLayer('loss3',EPELoss(),{'predict_flow3','label3'},'objective3');
net.addLayer('loss2',EPELoss(),{'predict_flow2','label2'},'objective2');

net.initParams();