function train_id_net_vgg16(varargin)
% -------------------------------------------------------------------------
% Part 4.1: prepare the data
% -------------------------------------------------------------------------

imdb = load('./dataset/MSCOCO-prepare/url_data.mat');
imdb = imdb.imdb;
load('./dataset/MSCOCO-prepare/coco_word2.mat');
%sort row
[imdb.images.label2,index] = sort(imdb.images.label2);
wordcnn = wordcnn(:,index);
imdb.charcnn = wordcnn; 
%imdb.charmean = mean(imdb.charcnn(:,:,:,imdb.images.set==1),4);
% -------------------------------------------------------------------------
% Part 4.2: initialize a CNN architecture
% -------------------------------------------------------------------------
net = coco_word2_Rankloss();
net.conserveMemory = true;
im_mean = imdb.rgbMean;
net.meta.normalization.averageImage = im_mean;
%net.meta.normalization.charmean = imdb.charmean;
% -------------------------------------------------------------------------
% Part 4.3: train and evaluate the CNN

% -------------------------------------------------------------------------
opts.train.averageImage = net.meta.normalization.averageImage;
opts.train.batchSize = 32;
opts.train.continue = true;
opts.train.gpus = 1;
opts.train.prefetch = false ;
opts.train.nesterovUpdate = true ;
opts.train.expDir = './data/res52_coco_batch32_Rankloss_2:1:0.1_margin1_both_drop0.5_hard_256';
opts.train.derOutputs = {'objective_f',2,'objective_img',1,'objective_txt',0.1} ;
%opts.train.gamma = 0.9;
opts.train.candidate = 256;
opts.train.momentum = 0.9;
%opts.train.constraint = 100;
opts.train.learningRate = [0.1*ones(1,15),0.01*ones(1,5),0.001] ;
opts.train.weightDecay = 0.0001;
opts.train.numEpochs = numel(opts.train.learningRate) ;
[opts, ~] = vl_argparse(opts.train, varargin) ;
% Call training function in MatConvNet
[net,info] = cnn_train_dag_batchsize_net(net, imdb, @getBatch,opts) ;

% --------------------------------------------------------------------
function inputs = getBatch(imdb,batch,net,opts)
% --------------------------------------------------------------------
%-- img data
im_url = imdb.images.data(batch) ;
im = vl_imreadjpeg(im_url,'Pack','Resize',[224,224],'Flip',...
    'CropLocation','random','CropSize',[0.8,1],...
    'Interpolation', 'bicubic','NumThreads',16,... %'Brightness', double(0.1*imdb.rgbCovariance),...
    'SubtractAverage',imdb.rgbMean,...
    'CropAnisotropy',[3/4,4/3]);
oim = im{1}; %bsxfun(@minus,im{1},opts.averageImage);
label_img =  imdb.images.label(batch);
label_txt = label_img;
% ------ first half img feature
net.mode = 'test';
net.vars(net.getVarIndex('fc2bn_n')).value = [];
net.vars(net.getVarIndex('data')).value = [];
net.vars(net.getVarIndex('data2')).value = [];
net.vars(net.getVarIndex('label_txt')).value = [];
net.vars(net.getVarIndex('label_img')).value = [];
%reset(net);
batchsize = numel(batch);
half = batchsize/2;
f_img = getFeature2(net,oim(:,:,:,1:half),[],'data','fc1bn_n');
f_img = reshape(f_img,[],half)';
%reset(net);

% ------- rand txt input  ---different with flickr30k
neg_label = imdb.images.label;
neg_label(batch(1:half))=[]; %remove same label
neg_label(neg_label==0) = []; % remove test and val images
neg_label = neg_label(randi(numel(neg_label), opts.candidate,1)); % random select 256 image sample
for i=1:opts.candidate
  rand_ind(i) = rand_same_class_coco(imdb,neg_label(i)); % neg txt index
end

txt = single(imdb.charcnn(:,rand_ind));
txtinput_rand = zeros(1,32,29972,opts.candidate,'single');
for i=1:opts.candidate
    len = sum(txt(:,i)>0);
    location = randi(33-len);
    for j=1:len
        v = txt(j,i);
        txtinput_rand(1,location,v,i)=1;
        location = location+1;
    end
end

%---- txt feature
net.vars(net.getVarIndex('fc1bn_n')).value = [];
f_txt = getFeature2(net,gpuArray(txtinput_rand),[],'data2','fc2bn_n');
size4 = size(f_txt,4);
f_txt = reshape(f_txt,[],size4)';
reset(net);
%-- txt data
txt_batch = zeros(1,half);
for i=1:half
  txt_batch(i) = rand_same_class_coco(imdb,label_img(i));
end

txt = single(imdb.charcnn(:,txt_batch));
txtinput = zeros(1,32,29972,batchsize,'single');
for i=1:half
    len = numel(find(txt(:,i)>0));
    location = randi(33-len);
    for j=1:len
        v = txt(j,i);
        txtinput(1,location,v,i)=1;
        location = location+1;
    end
end
%select hard txt sample
for i= half+1:batchsize
   [~,max_ind] = max(f_img(i-half,:)*f_txt'); 
    txtinput(:,:,:,i) = txtinput_rand(:,:,:,max_ind);
    label_txt(i) = neg_label(max_ind);
end
%reset(net);
%--
inputs = {'data',gpuArray(oim),'data2',gpuArray(txtinput),'label_img',label_img,'label_txt',label_txt};
