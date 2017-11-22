classdef RankLoss < dagnn.Loss
    properties
        rate = 0.5
    end
    methods
        function outputs = forward(obj, inputs, params)
            % dataX:inputs{1} 1*1*1024*32
            % dataY:inputs{2} 1*1*1024*32
            %----step1:resize data
            batchsz = size(inputs{2},4);
            dataX = reshape(inputs{1},[],batchsz);
            dataY = reshape(inputs{2},[],batchsz);
            
            %----step2:get
            half = batchsz/2;
            Xp =  dataX(:,1:half);
            Xn =  dataX(:,half+1:end);
            Yp =  dataY(:,1:half);
            Yn =  dataY(:,half+1:end);
            
            positive = diag(Xp'*Yp);
            Loss1 = max(0,obj.rate-(positive-diag(Xp'*Yn)));
            Loss2 = max(0,obj.rate-(positive-diag(Yp'*Xn)));
            
            outputs{1} = gpuArray(sum(Loss1(:))+sum(Loss2(:)));
            n = obj.numAveraged ;
            m = n + size(inputs{1},4) ;
            obj.average = (n * obj.average + gather(outputs{1})) / m ;
            obj.numAveraged = m ;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            %----step1:resize data
            batchsz = size(inputs{2},4);
            dataX = reshape(inputs{1},[],batchsz);
            dataY = reshape(inputs{2},[],batchsz);
            
            %----step2:get
            half = batchsz/2;
            Xp =  dataX(:,1:half);
            Xn =  dataX(:,half+1:end);
            Yp =  dataY(:,1:half);
            Yn =  dataY(:,half+1:end);
            
            positive = diag(Xp'*Yp);
            Loss1 = obj.rate-(positive-diag(Xp'*Yn));  %16*1
            Loss2 = obj.rate-(positive-diag(Yp'*Xn));  
            
            %-----
            derLoss1 = repmat(Loss1>0,1,size(inputs{1},3))';  %2048*16
            derLoss2 = repmat(Loss2>0,1,size(inputs{2},3))';
            
            dLoss1_dXp = (Yn-Yp).*derLoss1; %2048*16
            dLoss1_dYp = -Xp.*derLoss1;
            dLoss1_dYn = Xp.*derLoss1;
            
            dLoss2_dYp = (Xn-Xp).*derLoss2;
            dLoss2_dXp = -Yp.*derLoss2;
            dLoss2_dXn = Yp.*derLoss2;
            
            dX = cat(2,dLoss1_dXp+dLoss2_dXp,dLoss2_dXn);
            dY = cat(2,dLoss1_dYp+dLoss2_dYp,dLoss1_dYn);
            
            derInputs{1} = reshape(dX,1,1,size(inputs{1},3),size(inputs{1},4));
            derInputs{1} = gpuArray(bsxfun(@times,derOutputs{1},derInputs{1}));
            
            derInputs{2} = reshape(dY,1,1,size(inputs{2},3),size(inputs{2},4));
            derInputs{2} = gpuArray(bsxfun(@times,derOutputs{1},derInputs{2}));
            
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
        
        function obj = RankLoss(varargin)
            obj.load(varargin) ;
        end
    end
end
