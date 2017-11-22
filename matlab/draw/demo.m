clear;
%simplenn net
net = cnn_cifar_init();
net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
net.addLayer('error', dagnn.Loss('loss', 'classerror'), ...
    {'prediction','label'}, 'error') ;
draw_full_net(net,'cifar_net');

%dagnn net
%net =test_neta(); % 2_stream_inception_net
%draw_full_net(net,'test_neta');
net = flownet();
draw_full_net(net,'flownet');