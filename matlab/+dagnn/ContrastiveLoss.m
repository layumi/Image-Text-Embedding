classdef ContrastiveLoss < dagnn.Loss
  methods
    function outputs = forward(obj, inputs, params)
      % data:inputs{1} 
      % label:inputs{2}   same:1  diff:2
      y = 2-inputs{2};  % y same:1   diff:0
      batchsz = numel(inputs{2});
      x = reshape(inputs{1},1,batchsz); %1*48
      J =(x.^2*y + max((0.7-x),0).^2*(1-y))';
      outputs{1} = gpuArray(sum(J));
      n = obj.numAveraged ;
      m = n + size(inputs{1},4) ;
      obj.average = (n * obj.average + gather(outputs{1})) / m ;
      obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      y = 2-inputs{2};
      batchsz = numel(inputs{2});
      x = reshape(inputs{1},1,batchsz);
      dermax = (0.7-x)>0;
      derNorm2 = 2*(y'.*x - (1-y)'.*(0.7-x).*dermax); %1*48
      derInputs{1} = reshape(derNorm2,1,1,1,batchsz);
      derInputs{1} = gpuArray(bsxfun(@times,derOutputs{1},derInputs{1}));
      derInputs{2} = [];
      derParams = {} ;
    end

    function reset(obj)
      obj.average = 0 ;
      obj.numAveraged = 0 ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes, paramSizes)
      outputSizes{1} = [1 1 1 inputSizes{1}(4)] ;
    end

    function rfs = getReceptiveFields(obj)
      % the receptive field depends on the dimension of the variables
      % which is not known until the network is run
      rfs(1,1).size = [NaN NaN] ;
      rfs(1,1).stride = [NaN NaN] ;
      rfs(1,1).offset = [NaN NaN] ;
      rfs(2,1) = rfs(1,1) ;
    end

    function obj = ContrastiveLoss(varargin)
      obj.load(varargin) ;
    end
  end
end
