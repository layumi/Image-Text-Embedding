#include "Hybrid.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include "mex.h"
#include <omp.h>


using namespace std;

inline void getMixtureWeightsForSingleSample(int numOfModels, int numOfDims, fp_type* x, fp_type** m, fp_type** s, fp_type** mu, fp_type** sigma, bool** b, fp_type* priors, fp_type* mixtureWeightsOut)
{
	fp_type maxExp = -DBL_MAX;

	for (int k=0; k<numOfModels; k++)
	{
		mixtureWeightsOut[k] = getMultivariateHybridLogPDF(numOfDims, x, m[k], s[k], mu[k], sigma[k], b[k]);

		if (mixtureWeightsOut[k] > maxExp)	
		{
			maxExp = mixtureWeightsOut[k];
		}
	}
	
	/* for numeric reason */
	fp_type denominator = 0.0;
	for (int k=0; k<numOfModels; k++)
	{
		mixtureWeightsOut[k] = mixtureWeightsOut[k] - maxExp;
		denominator = denominator + priors[k]*exp(mixtureWeightsOut[k]);
	}

	for (int k=0; k<numOfModels; k++)
	{
		mixtureWeightsOut[k] = priors[k]*exp(mixtureWeightsOut[k]) / denominator;
	}
}

void computeCurrentMixtureWeightsPara(int numOfSamples, int numOfDims, int numOfModels, fp_type** samples, fp_type** m, fp_type** s, fp_type** mu, fp_type** sigma, bool** b, fp_type* priors, fp_type** outMixtureWeights, fp_type* outTotalMixtureWeight)
{
	#pragma omp parallel for
	for (int k=0; k<numOfModels; k++)
	{
		outTotalMixtureWeight[k] = 0.0;
	}

	#pragma omp parallel for
	for (int i=0; i<numOfSamples; i++)
	{
		getMixtureWeightsForSingleSample(numOfModels, numOfDims, samples[i], m, s, mu, sigma, b, priors, outMixtureWeights[i]);
	}
	
	#pragma omp parallel for
	for (int k=0; k<numOfModels; k++)
	{
		for (int i=0; i < numOfSamples; i++)
		{
			outTotalMixtureWeight[k] = outTotalMixtureWeight[k] + outMixtureWeights[i][k];
		}
	}
}

template <typename T>
vector<size_t> sort_indexes(const vector<T> &v) {

  // initialize original index locations
  vector<size_t> idx(v.size());
  for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;

  // sort indexes based on comparing values in v
  sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

  return idx;
}

void updateMixureBParaSingleDim(int numOfSamples, int numOfDims, int numOfModels, fp_type** samples, fp_type** currentMixtureWeights, fp_type* totalMixtureWeight, fp_type** currentM, fp_type** currentS, fp_type** currentMu, fp_type** currentSigma, bool** outB, int relevantDim)
{
	for (int k=0; k<numOfModels; k++)
	{
		fp_type accumL = 0.0;
		fp_type accumG = 0.0;
		
		for (int i=0; i < numOfSamples; i++)
		{
			accumL = accumL + (currentMixtureWeights[i][k] * getUnivariateLaplaceLogPDF(samples[i][relevantDim], currentM[k][relevantDim], currentS[k][relevantDim]));
			accumG = accumG + (currentMixtureWeights[i][k] * getUnivariateGaussianLogPDF(samples[i][relevantDim], currentMu[k][relevantDim], currentSigma[k][relevantDim]));
		}
		
		if (accumL > accumG)
		{
			outB[k][relevantDim] = 1;
		}
		else
		{
			outB[k][relevantDim] = 0;
		}
	}
}

void updateMixureBPara(int numOfSamples, int numOfDims, int numOfModels, fp_type** samples, fp_type** currentMixtureWeights, fp_type* totalMixtureWeights, fp_type** currentM, fp_type** currentS, fp_type** currentMu, fp_type** currentSigma, bool** outB)
{
	#pragma omp parallel for
	for (int d=0; d < numOfDims; d++)
	{
		updateMixureBParaSingleDim(numOfSamples, numOfDims, numOfModels, samples, currentMixtureWeights, totalMixtureWeights, currentM, currentS, currentMu, currentSigma, outB, d);
	}
}

void updateMixtureSigmaSingleDim(int numOfSamples, int numOfDims, int numOfModels, fp_type** samples, fp_type** currentMixtureWeights, fp_type** newMu, fp_type* totalMixtureWeight, fp_type** outNewB, int relevantDim, fp_type minSigmaValue)
{
	for (int k=0; k<numOfModels; k++)
	{
			outNewB[k][relevantDim] = 0.0;

			for (int i=0; i < numOfSamples; i++)
			{
				outNewB[k][relevantDim] = outNewB[k][relevantDim] + currentMixtureWeights[i][k]*((samples[i][relevantDim] - newMu[k][relevantDim])*(samples[i][relevantDim] - newMu[k][relevantDim]));;
			}

			outNewB[k][relevantDim] = outNewB[k][relevantDim] / totalMixtureWeight[k];
			outNewB[k][relevantDim] = max(minSigmaValue, outNewB[k][relevantDim]);
	}
}


void updateMixtureSigmaPara(int numOfSamples, int numOfDims, int numOfModels, fp_type** samples, fp_type** currentMixtureWeights, fp_type** newMu, fp_type* totalMixtureWeight, fp_type** outNewB, fp_type minSigmaValue)
{
	#pragma omp parallel for
	for (int d=0; d < numOfDims; d++)
	{
		updateMixtureSigmaSingleDim(numOfSamples, numOfDims, numOfModels, samples, currentMixtureWeights, newMu, totalMixtureWeight, outNewB, d, minSigmaValue);
	}
}

void updateMixtureMuSingleDim(int numOfSamples, int numOfDims, int numOfModels, fp_type** samples, fp_type** currentMixtureWeights, fp_type* totalMixtureWeight, fp_type** outNewMu, int relevantDim)
{
	for (int k=0; k<numOfModels; k++)
	{			
		fp_type accum = 0.0;
			
		for (int i = 0; i < numOfSamples; i++)
		{
			accum = accum + currentMixtureWeights[i][k] * samples[i][relevantDim];
		}

		outNewMu[k][relevantDim] = accum / totalMixtureWeight[k];
	}
}

void updateMixtureMuPara(int numOfSamples, int numOfDims, int numOfModels, fp_type** samples, fp_type** currentMixtureWeights, fp_type* totalMixtureWeight, fp_type** outNewMu)
{
	#pragma omp parallel for
	for (int d=0; d < numOfDims; d++)
	{
		updateMixtureMuSingleDim(numOfSamples, numOfDims, numOfModels, samples, currentMixtureWeights, totalMixtureWeight, outNewMu, d);
	}
}

void updateMixtureMSingleDim(int numOfSamples, int numOfDims, int numOfModels, fp_type** samples, fp_type** currentMixtureWeights, fp_type* totalMixtureWeight, fp_type** outNewMu, int relevantDim)
{
	vector<fp_type> currentDimSamplesValue(numOfSamples);

	for (int i=0; i < numOfSamples; i++)
	{
		currentDimSamplesValue[i] = samples[i][relevantDim];
	}
		
	vector<size_t> sortedIdx = sort_indexes(currentDimSamplesValue);

	for (int k=0; k<numOfModels; k++)
	{
		int currentIdx;	
			
		fp_type totalAccumWeight = 0.0;
			
		for (int i = 0; i < numOfSamples; i++)
		{
			currentIdx = sortedIdx[i];
			totalAccumWeight = totalAccumWeight + currentMixtureWeights[currentIdx][k];
				
			if (totalAccumWeight >= (((fp_type) totalMixtureWeight[k]) / 2.0))
			{
				break;
			}
		}

		outNewMu[k][relevantDim] = samples[currentIdx][relevantDim];
	}
	
}

void updateMixtureMPara(int numOfSamples, int numOfDims, int numOfModels, fp_type** samples, fp_type** currentMixtureWeights, fp_type* totalMixtureWeight, fp_type** outNewMu)
{
	#pragma omp parallel for
	for (int d=0; d < numOfDims; d++)
	{
		updateMixtureMSingleDim(numOfSamples, numOfDims, numOfModels, samples, currentMixtureWeights, totalMixtureWeight, outNewMu, d);
	}
}

void updateMixtureSSingleDim(int numOfSamples, int numOfDims, int numOfModels, fp_type** samples, fp_type** currentMixtureWeights, fp_type** newMu, fp_type* totalMixtureWeight, fp_type** outNewB, int relevantDim, fp_type minSValue)
{
	for (int k=0; k<numOfModels; k++)
	{
			outNewB[k][relevantDim] = 0.0;

			for (int i=0; i < numOfSamples; i++)
			{
				outNewB[k][relevantDim] = outNewB[k][relevantDim] + currentMixtureWeights[i][k]*abs(samples[i][relevantDim] - newMu[k][relevantDim]);
			}

			outNewB[k][relevantDim] = outNewB[k][relevantDim] / totalMixtureWeight[k];
			outNewB[k][relevantDim] = max(minSValue, outNewB[k][relevantDim]);
	}
}


void updateMixtureSPara(int numOfSamples, int numOfDims, int numOfModels, fp_type** samples, fp_type** currentMixtureWeights, fp_type** newMu, fp_type* totalMixtureWeight, fp_type** outNewB, fp_type minSValue)
{
	#pragma omp parallel for
	for (int d=0; d < numOfDims; d++)
	{
		updateMixtureSSingleDim(numOfSamples, numOfDims, numOfModels, samples, currentMixtureWeights, newMu, totalMixtureWeight, outNewB, d, minSValue);
	}
}

void updateMixturePriors(int numOfSamples, int numOfModels, fp_type* totalMixtureWeight, fp_type* outPriors)
{
	for (int k=0; k < numOfModels; k++)
	{
		outPriors[k] = totalMixtureWeight[k] / numOfSamples;
	}
}

void lmmEM(int numOfSamples, int numOfModels, int numOfDims, int max_iterations, fp_type** samples, fp_type** init_m, fp_type** init_s, fp_type** init_mu, fp_type** init_sigma, bool** init_b, fp_type* init_priors, fp_type** out_final_m, fp_type** out_final_s, fp_type** out_final_mu, fp_type** out_final_sigma, bool** out_final_b, fp_type* out_final_prior, int* out_num_of_iter, fp_type** out_final_mixture_weight, int numOfCores, fp_type min_s, fp_type min_sigma)
{
	omp_set_num_threads(numOfCores);

	fp_type** currentM;
	fp_type** currentS;
	fp_type** currentMu;
	fp_type** currentSigma;
	bool** currentB;
	
	fp_type* currentPriors;
	
	
	
	
	fp_type** currentMixtureWeights;
	fp_type** tempMixtureWeights;
	fp_type* totalMixtureWeights;

	currentPriors = new fp_type[numOfModels];

	totalMixtureWeights = new fp_type[numOfModels];

	currentM = new fp_type*[numOfModels];
	currentS = new fp_type*[numOfModels];
	currentMu = new fp_type*[numOfModels];
	currentSigma = new fp_type*[numOfModels];
	currentB = new bool*[numOfModels];

	currentMixtureWeights = new fp_type*[numOfSamples];

	fp_type currentSamplesLogLikelihood;

	for (int k=0; k < numOfModels; k++)
	{
		currentM[k] = new fp_type[numOfDims];
		currentS[k] = new fp_type[numOfDims];
		currentMu[k] = new fp_type[numOfDims];
		currentSigma[k] = new fp_type[numOfDims];
		currentB[k] = new bool[numOfDims];
		
		for (int d=0; d < numOfDims; d++)
		{
			currentM[k][d] = init_m[k][d];
			currentS[k][d] = init_s[k][d];
			currentMu[k][d] = init_mu[k][d];
			currentSigma[k][d] = init_sigma[k][d];
			currentB[k][d] = init_b[k][d];
		}

		currentPriors[k] = init_priors[k];
	}

	for (int i=0; i < numOfSamples; i++)
	{
		currentMixtureWeights[i] = new fp_type[numOfModels];
	}

	int iter;

	for (iter=0; iter < max_iterations; iter++)
	{
		currentSamplesLogLikelihood = getSamplesMixtureLogLikelihood(numOfModels, numOfSamples, numOfDims, samples, currentM, currentS, currentMu, currentSigma, currentB, currentPriors);
		
		mexPrintf("Current Likelihood %f\n",currentSamplesLogLikelihood);
		mexEvalString("drawnow");

		mexPrintf("Starting iteration #%d",iter);
		
		computeCurrentMixtureWeightsPara(numOfSamples, numOfDims, numOfModels, samples, currentM, currentS, currentMu, currentSigma, currentB, currentPriors, currentMixtureWeights, totalMixtureWeights);
		
		fp_type diffMixtureWeightsNorm = 0.0;

		mexPrintf("\n");

		updateMixtureMPara(numOfSamples, numOfDims, numOfModels, samples, currentMixtureWeights, totalMixtureWeights, currentM); // update laplacian M
		updateMixtureSPara(numOfSamples, numOfDims, numOfModels, samples, currentMixtureWeights, currentM, totalMixtureWeights, currentS, min_s); // update laplacian S
		updateMixtureMuPara(numOfSamples, numOfDims, numOfModels, samples, currentMixtureWeights, totalMixtureWeights, currentMu); // update gaussian Mu
		updateMixtureSigmaPara(numOfSamples, numOfDims, numOfModels, samples, currentMixtureWeights, currentMu, totalMixtureWeights, currentSigma, min_sigma); // update gaussian Sigma
		updateMixturePriors(numOfSamples, numOfModels, totalMixtureWeights, currentPriors); // update priors
		updateMixureBPara(numOfSamples, numOfDims, numOfModels, samples, currentMixtureWeights, totalMixtureWeights, currentM, currentS, currentMu, currentSigma, currentB); // update gaussian laplacian flag
	}
	
	currentSamplesLogLikelihood = getSamplesMixtureLogLikelihood(numOfModels, numOfSamples, numOfDims, samples, currentM, currentS, currentMu, currentSigma, currentB, currentPriors);
	
	mexPrintf("Current Likelihood %f\n",currentSamplesLogLikelihood);
	mexEvalString("drawnow");

	for (int k=0; k < numOfModels; k++)
	{
		for (int d=0; d<numOfDims; d++)
		{
			out_final_m[k][d] = currentM[k][d];
			out_final_s[k][d] = currentS[k][d];
			out_final_mu[k][d] = currentMu[k][d];
			out_final_sigma[k][d] = currentSigma[k][d];
			out_final_b[k][d] = currentB[k][d];
		}

		out_final_prior[k] = currentPriors[k];
	}

	out_num_of_iter[0] = iter; 

	for (int i=0; i < numOfSamples; i++)
	{
		for (int k=0; k < numOfModels; k++)
		{
			out_final_mixture_weight[i][k] = currentMixtureWeights[i][k];
		}
	}
		
	delete[] currentPriors;
	delete[] totalMixtureWeights;

	for (int k=0; k < numOfModels; k++)
	{
		delete[] currentM[k]; 
		delete[] currentS[k];
		delete[] currentMu[k];
		delete[] currentSigma[k];
		delete[] currentB[k];
	}

	delete[] currentM;
	delete[] currentS;
	delete[] currentMu;
	delete[] currentSigma;
	delete[] currentB;

	for (int i=0; i < numOfSamples; i++)
	{
		delete[] currentMixtureWeights[i];
	}

	delete[] currentMixtureWeights;
}

