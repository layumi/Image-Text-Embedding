to compile:
---------------
mex HybridEMMex.cpp Hybrid.cpp Util.cpp COMPFLAGS="/openmp $COMPFLAGS" CXXFLAGS="\$CXXFLAGS -std=c++0x" LDFLAGS="\$LDFLAGS -fopenmp"
mex HybridFVMex.cpp FV.cpp Hybrid.cpp Util.cpp COMPFLAGS="/openmp $COMPFLAGS" CXXFLAGS="\$CXXFLAGS -std=c++0x" LDFLAGS="\$LDFLAGS -fopenmp"


Usage examples:
---------------
EM
---

[ m_out, s_out, mu_out, sigma_out, b_out, prior_out, samplesWeightsOut ]=HybridEM(vectors,classes,max_iter,4,1e-3,1e-3,prior_init,m_init,s_init,mu_init,sigma_init,b_init);

Where:
vectors is #samples * #features matrix.
classes are the number of clusters that we want in the mixture.
max_iter - The number of EM iterations
4 - The number of cpu cores that will be used for the EM
1e-3 - The minimal value of a S parameter (You can leave it like that)
1e-3 - The minimal value of a Sigma parameter (You can leave it like that)
prior_init - The initial guess for the prior of each component in the mixture
m_init - the initial guess for the mean of the laplacian
s_init - the initial guess for the std of the laplacian
mu_init - the initial guess for the mean of the gaussian
sigma_init - the initial guess for the var of the gaussian
b_init - the initial guess for the associate of each component-dimension to gaussian or laplacian distribution (This is a logical value)

Only the first 4 parameters are a must. Therefore you can call: HybridEM(vectors,classes,max_iter,4) and the function will handle the value of the other parameters.

FV
---

fisherHybrid = HybridFV(double(vectors), m_out, s_out, mu_out, sigma_out, b_out, prior_out, 4);

Where 4 is the number of cpu cores that will be used for the FV.


One annoying issue:
-------------------
If vectors contains small enough values, numerical issues may arise. 
The effect of this problem, is that the EM does not converge. If it happens to you, then a quick solution would be:
Before running the HybridEM (And before computing its initial values), multiply the vectors matrix by a constant (say 10). This will make the values large enough for the HybridEM to work. If you do this, then you should do the same before running HybridFV, meaning that vectors should also be multiplied by the same constant.
