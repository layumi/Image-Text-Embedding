to compile:
---------------
mex LMMMex.cpp LMM.cpp UtilLMM.cpp COMPFLAGS="/openmp $COMPFLAGS"
mex FVMex.cpp FV.cpp LMM.cpp UtilLMM.cpp COMPFLAGS="/openmp $COMPFLAGS"

Usage examples:
---------------
EM:
---
the GMM (from VLFEAT) does a random init, which our mex does not. Therefore, you should create a random initialization before calling it:

initPriors = (1.0 / numClusters) * ones(numClusters,1);
muInit=vectors(randperm(size(vectors,1),numClusters),:);
init_b=repmat(sum(abs(vectors - repmat(median(vectors),size(vectors,1),1))) / size(vectors,1),numClusters,1);

Where numClusters is the number of clusters.
And vectors is #samples * #features matrix.

Now you can call the mex:

numOfEMIter = 10;
[n_means,n_covariances,n_priors,samplesWeightsOut]=LMMMex(double(vectors), double(muInit), double(init_b), double(initPriors), numOfEMIter, 0.01, 2);

numOfEMIter is the number of EM iterations that the algorithm will do before returning the output. Suggested value: 10 (You will see that with 10 iterations, it will converge and there is no need for more iterations).
After each iteration, you will get the log-likelihood as an output. It should be negative, and should be increasing in every iteration.

The 0.01 is a constant that should stay as is.
The 2 is number of cpu cores to be used.

FV
---
encodingRes = FVMex(double(wordsToEncode),double(gmm_means), double(gmm_covariances), double(gmm_priors),1.0);

The 1.0 is a constant that should stay as is.

Please notice that FVMex does not take sqrtRoot and does not normalize by L2-norm. Therefore you can use:

if (sqrtRootFlag)
    encodingRes = sign(encodingRes) .* sqrt(abs(encodingRes)); 
end
                
encoding = encodingRes / norm(encodingRes);


One annoying issue:
-------------------
If wordsToEncode contains small enough values, numerical issues may arise. 
The effect of this problem, is that the EM does not converge. If it happens to you, then a quick solution would be:
Before running the LMMMex (And before computing its initial values), multiply the vectors matrix by a constant (say 10). This will make the values large enough for the LMMMex to work. If you do this, then you should do the same before running FVMex, meaning that wordsToEncode should also be multiplied by the same constant.
