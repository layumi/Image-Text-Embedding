function train_id_net_vgg16(varargin)
% -------------------------------------------------------------------------
% Part 4.1: prepare the data
% -------------------------------------------------------------------------

imdb = load('url_data.mat');
imdb = imdb.imdb;
load('./Flickr30k/dense_feature_word2.1.mat');
imdb.charcnn = wordcnn;
%imdb.charmean = mean(imdb.charcnn(:,:,:,imdb.images.set==1),4);
% -------------------------------------------------------------------------
% Part 4.2: initialize a CNN architecture
% -------------------------------------------------------------------------
net = resnet52_new_hope_word2_pool_ft();
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
opts.train.gpus = 4;
opts.train.prefetch = false ;
opts.train.nesterovUpdate = true ;
opts.train.expDir = './data/res52_batch32_new_hope_word2.1_pool_shift_both_drop0.75_ft';
opts.train.derOutputs = {'objective_img',1,'objective_txt',1} ;
%opts.train.gamma = 0.9;
opts.train.momentum = 0.9;
%opts.train.constraint = 100;
opts.train.learningRate = [0.1*ones(1,240)] ;
opts.train.weightDecay = 0.0001;
opts.train.numEpochs = numel(opts.train.learningRate) ;
[opts, ~] = vl_argparse(opts.train, varargin) ;
% Call training function in MatConvNet
[net,info] = cnn_train_dag(net, imdb, @getBatch,opts) ;

% --------------------------------------------------------------------
function inputs = getBatch(imdb,batch,opts)
% --------------------------------------------------------------------
%-- img data
im_url = imdb.images.data(batch) ;
im = vl_imreadjpeg(im_url,'Pack','Resize',[224,224],'Flip',...
    'CropLocation','random','CropSize',[0.8,1],...
    'Interpolation', 'bicubic','NumThreads',24,... %'Brightness', double(0.1*imdb.rgbCovariance),...
    'SubtractAverage',imdb.rgbMean,...
    'CropAnisotropy',[3/4,4/3]);
oim = im{1}; %bsxfun(@minus,im{1},opts.averageImage);
label_img =  imdb.images.label(batch);

%-- txt data
batchsize = numel(batch);
label_txt =  imdb.images.label(batch);
txt_branch = bsxfun(@times,(batch-1),5) + randi(5,batchsize,1);
txt = single(imdb.charcnn(:,txt_branch));
% code for faster construct one-hot vector (remove loop)   but seems not
% faster.... so I give up it.
%{
txt_v = txt(:);
txt_v(:,2) = reshape(repmat(0:(batchsize-1),32,1),1,[])';
txt_v(:,3) = reshape(repmat(0:31,1,batchsize),1,[])';
txt_v = txt_v(txt_v(:,1)>0,:);
txt_v = txt_v(:,1)+ bsxfun(@times,txt_v(:,2),642368)... %20074*32
    +bsxfun(@times,txt_v(:,3),20074);
txtinput = gpuArray(zeros(1,20074,32,batchsize,'single'));
txtinput(txt_v) = 1;
txtinput = permute(txtinput,[1,3,2,4]);
%}
txtinput = zeros(1,32,20074,batchsize,'single');
for i=1:batchsize
    len = numel(find(txt(:,i)>0));
    location = randi(33-len);
    for j=1:len
        v = txt(j,i);
        txtinput(1,location,v,i)=1;
        location = location+1;
    end
end
txtinput = gpuArray(txtinput);
%}
%--
inputs = {'data',gpuArray(oim),'data2',txtinput,'label_img',label_img,'label_txt',label_txt};
