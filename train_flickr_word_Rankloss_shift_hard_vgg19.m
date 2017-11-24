function train_id_net_vgg16(varargin)
% -------------------------------------------------------------------------
% Part 4.1: prepare the data
% -------------------------------------------------------------------------

imdb = load('./dataset/Flickr30k-prepare/url_data.mat');
imdb = imdb.imdb;
load('./dataset/Flickr30k-prepare/dense_feature_word2.1.mat');
imdb.charcnn = wordcnn; 
%imdb.charmean = mean(imdb.charcnn(:,:,:,imdb.images.set==1),4);
% -------------------------------------------------------------------------
% Part 4.2: initialize a CNN architecture
% -------------------------------------------------------------------------
net = resnet52_new_hope_word_Rankloss_vgg19();
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
opts.train.gpus = 2;
opts.train.prefetch = false ;
opts.train.nesterovUpdate = true ;
opts.train.expDir = './data/res52_batch32_Rankloss_2:1:0.1_margin1_img0.75_shift_hard_256_vgg19';
opts.train.derOutputs = {'objective_f',2,'objective_img',1,'objective_txt',0.1} ;
opts.train.candidate = 256;
%opts.train.gamma = 0.9;
opts.train.momentum = 0.9;
%opts.train.constraint = 100;
opts.train.learningRate = [0.1*ones(1,40),0.01*ones(1,20)] ;
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
oim = gpuArray(im{1}); 
label_img =  imdb.images.label(batch);

%----- first half img feature
net.mode = 'test' ;
%net.vars(net.getVarIndex('fc2bn_n')).value = [];
%net.vars(net.getVarIndex('data')).value = [];
%net.vars(net.getVarIndex('data2')).value = [];
%net.vars(net.getVarIndex('label_img')).value = [];
%net.vars(net.getVarIndex('label_txt')).value = [];
reset(net);
batchsize = numel(batch);
half = batchsize/2;
f_img = getFeature2(net,oim(:,:,:,1:half),[],'x0','fc1bn_n');
f_img = reshape(f_img,[],half)';

%------ rand txt input
ind = imdb.images.set==1;  % training set
ind(batch) = 0;  % remove the same label
ind = find(ind==1);  % change 0-1 to real index
ind = (ind-1)*5 + randi(5,numel(ind),1); % img index to char index
rand_ind = ind(randi(numel(ind),opts.candidate,1)); % random select 256 sample

txt = single(imdb.charcnn(:,rand_ind));
txtinput_rand = zeros(1,32,20074,opts.candidate,'single');
for i=1:opts.candidate
    len = sum(txt(:,i)>0);
    location = randi(33-len);
    for j=1:len
        v = txt(j,i);
        txtinput_rand(1,location,v,i)=1;
        location = location+1;
    end
end

%---txt feature
net.vars(net.getVarIndex('fc1bn_n')).value = [];
f_txt = getFeature2(net,gpuArray(txtinput_rand),[],'data2','fc2bn_n');
size4 = size(f_txt,4);
f_txt = reshape(f_txt,[],size4)';

txt_branch = bsxfun(@times,(batch-1),5) + randi(5,batchsize,1); %correct sample
txt = single(imdb.charcnn(:,txt_branch));
txtinput = zeros(1,32,20074,batchsize,'single');
% correct half
for i=1:half
    len = sum(txt(:,i)>0);
    location = randi(33-len);
    for j=1:len
       v = txt(j,i);
       txtinput(1,location,v,i)=1;
       location = location+1;
    end
end
%--- select hard txt sample 
for i= half+1:batchsize
    [~,max_ind] = max(f_img(i-half,:)*f_txt'); %1x2048 2048x256
    txt_branch(i) = rand_ind(max_ind);
    txtinput(:,:,:,i) = txtinput_rand(:,:,:,max_ind);
end
label_txt =  imdb.images.label(floor((txt_branch-1)/5)+1);

%----- reset net
net.vars(net.getVarIndex('fc1bn_n')).value = [];
net.vars(net.getVarIndex('fc2bn_n')).value = [];
net.mode = 'normal' ;

%--
inputs = {'x0',oim,'data2',gpuArray(txtinput),'label_img',label_img,'label_txt',label_txt};
