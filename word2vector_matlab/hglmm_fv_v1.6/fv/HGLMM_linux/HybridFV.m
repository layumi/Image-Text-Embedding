function [ fvRep ] = HybridFV( points, m, s, mu, sigma, b, prior, numOfCores )
fvRep = HybridFVMex(double(points), m, s, mu, sigma, b, prior, logical(1), numOfCores);
end

