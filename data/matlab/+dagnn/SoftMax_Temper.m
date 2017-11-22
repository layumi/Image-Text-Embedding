classdef SoftMax_Temper < dagnn.ElementWise
  properties
    temper = 1
  end

  methods
    function outputs = forward(self, inputs, params)
      outputs{1} = vl_nnsoftmax_temper(inputs{1},self.temper) ;
    end

    function [derInputs, derParams] = backward(self, inputs, params, derOutputs)
      derInputs{1} = vl_nnsoftmax(inputs{1}, self.temper,derOutputs{1}) ;
      fprintf('\nI am lazy...\n');
      derParams = {} ;
    end

    function obj = SoftMax_Temper(varargin)
      obj.load(varargin) ;
    end
  end
end
