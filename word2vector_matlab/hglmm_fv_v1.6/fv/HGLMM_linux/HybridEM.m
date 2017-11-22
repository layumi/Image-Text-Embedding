function [ m_out, s_out, mu_out, sigma_out, b_out, prior_out, samplesWeightsOut ] = HybridEM( samples, numClusters, max_iter, numCores, minSValue, minSigmaValue, prior_init, m_init, s_init, mu_init, sigma_init, b_init )

if ~exist('prior_init','var')
    prior_init = (1.0 / numClusters) * ones(numClusters,1);
end

if ~exist('m_init','var')
    m_init=samples(randsample(size(samples,1),numClusters),:);
end

if ~exist('mu_init','var')
    mu_init=m_init;
end

if ~exist('s_init','var')
    s_init=repmat(sum(abs(samples - repmat(median(samples),size(samples,1),1))) / size(samples,1),numClusters,1);
end

if ~exist('sigma_init','var')
    sigma_init=repmat(sum((samples - repmat(mean(samples),size(samples,1),1)).^2) / size(samples,1),numClusters,1);
end

if ~exist('b_init','var')
    b_init = randi(2,numClusters,size(samples,2))-1;
end

if ~exist('minSValue','var')
    minSValue = 1e-3;
end

if ~exist('minSigmaValue','var')
    minSigmaValue = 1e-3;
end

[m_out, s_out, mu_out, sigma_out, b_out, prior_out, samplesWeightsOut] = HybridEMMex( samples, m_init, s_init, mu_init, sigma_init, b_init, prior_init, max_iter, numCores, minSValue, minSigmaValue );

end

