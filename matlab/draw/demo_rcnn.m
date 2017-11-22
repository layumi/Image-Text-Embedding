clear;
net = load('/home/zzd/re_ID_beta23/examples/fast_rcnn/fast-rcnn-vgg16-pascal07-dagnn.mat') ;
net = dagnn.DagNN.loadobj(net);
draw_full_net(net,'fast_rcnn');

