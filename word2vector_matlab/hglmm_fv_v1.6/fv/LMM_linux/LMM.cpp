#include "LMM.h"
#include "UtilLMM.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include "mex.h"

#include <omp.h>


using namespace std;

#define MIN_B 10e-6

void computeCurrentMixtureWeights(int numOfSamples, int numOfDims, int numOfModels, fp_type** samples, fp_type** mu, fp_type** b, fp_type* priors, fp_type** outMixtureWeights, fp_type* outTotalMixtureWeight)
{
	for (int k=0; k<numOfModels; k++)
	{
		outTotalMixtureWeight[k] = 0.0;
	}

	for (int i=0; i<numOfSamples; i++)
	{
		getMixtureWeightsForSingleSample(numOfModels, numOfDims, samples[i], mu, b, priors, outMixtureWeights[i]);

		for (int k=0; k<numOfModels; k++)
		{
			outTotalMixtureWeight[k] = outTotalMixtureWeight[k] + outMixtureWeights[i][k];
		}
	}
}

void computeCurrentMixtureWeightsPara(int numOfSamples, int numOfDims, int numOfModels, fp_type** samples, fp_type** mu, fp_type** b, fp_type* priors, fp_type** outMixtureWeights, fp_type* outTotalMixtureWeight)
{
	#pragma omp parallel for
	for (int k=0; k<numOfModels; k++)
	{
		outTotalMixtureWeight[k] = 0.0;
	}

	#pragma omp parallel for
	for (int i=0; i<numOfSamples; i++)
	{
		getMixtureWeightsForSingleSample(numOfModels, numOfDims, samples[i], mu, b, priors, outMixtureWeights[i]);
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

void updateMixtureMuSingleDim(int numOfSamples, int numOfDims, int numOfModels, fp_type** samples, fp_type** currentMixtureWeights, fp_type* totalMixtureWeight, fp_type** outNewMu, int relevantDim)
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

void updateMixtureMuPara(int numOfSamples, int numOfDims, int numOfModels, fp_type** samples, fp_type** currentMixtureWeights, fp_type* totalMixtureWeight, fp_type** outNewMu)
{
	#pragma omp parallel for
	for (int d=0; d < numOfDims; d++)
	{
		updateMixtureMuSingleDim(numOfSamples, numOfDims, numOfModels, samples, currentMixtureWeights, totalMixtureWeight, outNewMu, d);
	}
}

void updateMixtureMu(int numOfSamples, int numOfDims, int numOfModels, fp_type** samples, fp_type** currentMixtureWeights, fp_type* totalMixtureWeight, fp_type** outNewMu)
{
	vector<fp_type> currentDimSamplesValue(numOfSamples);

	for (int d=0; d < numOfDims; d++)
	{
		for (int i=0; i < numOfSamples; i++)
		{
			currentDimSamplesValue[i] = samples[i][d];
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

			outNewMu[k][d] = samples[currentIdx][d];
		}
	}
}


void updateMixtureBSingleDim(int numOfSamples, int numOfDims, int numOfModels, fp_type** samples, fp_type** currentMixtureWeights, fp_type** newMu, fp_type* totalMixtureWeight, fp_type** outNewB, int relevantDim)
{
	for (int k=0; k<numOfModels; k++)
	{
			outNewB[k][relevantDim] = 0.0;

			for (int i=0; i < numOfSamples; i++)
			{
				outNewB[k][relevantDim] = outNewB[k][relevantDim] + currentMixtureWeights[i][k]*abs(samples[i][relevantDim] - newMu[k][relevantDim]);
			}

			outNewB[k][relevantDim] = outNewB[k][relevantDim] / totalMixtureWeight[k];
			outNewB[k][relevantDim] = max(MIN_B, outNewB[k][relevantDim]);
	}
}


void updateMixtureBPara(int numOfSamples, int numOfDims, int numOfModels, fp_type** samples, fp_type** currentMixtureWeights, fp_type** newMu, fp_type* totalMixtureWeight, fp_type** outNewB)
{
	#pragma omp parallel for
	for (int d=0; d < numOfDims; d++)
	{
		updateMixtureBSingleDim(numOfSamples, numOfDims, numOfModels, samples, currentMixtureWeights, newMu, totalMixtureWeight, outNewB, d);
	}
}

void updateMixtureB(int numOfSamples, int numOfDims, int numOfModels, fp_type** samples, fp_type** currentMixtureWeights, fp_type** newMu, fp_type* totalMixtureWeight, fp_type** outNewB)
{
	for (int k=0; k<numOfModels; k++)
	{
		for (int d=0; d<numOfDims; d++)
		{
			outNewB[k][d] = 0.0;

			for (int i=0; i < numOfSamples; i++)
			{
				outNewB[k][d] = outNewB[k][d] + currentMixtureWeights[i][k]*abs(samples[i][d] - newMu[k][d]);
			}

			outNewB[k][d] = outNewB[k][d] / totalMixtureWeight[k];
		}
	}
}

void updateMixturePriors(int numOfSamples, int numOfModels, fp_type* totalMixtureWeight, fp_type* outPriors)
{
	for (int k=0; k < numOfModels; k++)
	{
		outPriors[k] = totalMixtureWeight[k] / numOfSamples;
	}
}

void lmmEM(int numOfSamples, int numOfModels, int numOfDims, int max_iterations, fp_type thres, fp_type** samples, fp_type** init_mu, fp_type** init_b, fp_type* init_priors, fp_type** out_final_mu, fp_type** out_final_b, fp_type* out_final_prior, int* out_num_of_iter, fp_type** out_final_mixture_weight, int numOfCores)
{
	omp_set_num_threads(numOfCores);

	fp_type** firstMu; 
	fp_type** firstB; 
	fp_type* firstPriors; 

	fp_type** secondMu; 
	fp_type** secondB; 
	fp_type* secondPriors; 

	fp_type** currentMixtureWeights;
	fp_type** tempMixtureWeights;

	fp_type* totalMixtureWeights;

	firstPriors = new fp_type[numOfModels];
	secondPriors = new fp_type[numOfModels];
	totalMixtureWeights = new fp_type[numOfModels];

	firstMu = new fp_type*[numOfModels];
	secondMu = new fp_type*[numOfModels];
	firstB = new fp_type*[numOfModels];
	secondB = new fp_type*[numOfModels];

	currentMixtureWeights = new fp_type*[numOfSamples];
	tempMixtureWeights = new fp_type*[numOfSamples];

	fp_type oldSamplesLogLikelihood;
	fp_type newSamplesLogLikelihood;

	for (int k=0; k < numOfModels; k++)
	{
		firstMu[k] = new fp_type[numOfDims];
		secondMu[k] = new fp_type[numOfDims];
		firstB[k] = new fp_type[numOfDims];
		secondB[k] = new fp_type[numOfDims];

		for (int d=0; d < numOfDims; d++)
		{
			firstMu[k][d] = init_mu[k][d];
			firstB[k][d] = init_b[k][d];
		}

		firstPriors[k] = init_priors[k];
	}

	/* this will make life easier when switching between old and new */
	fp_type** old_priors;
	fp_type** new_priors;

	fp_type*** old_mu;
	fp_type*** new_mu;

	fp_type*** old_b;
	fp_type*** new_b;

	old_priors = &firstPriors;
	old_mu = &firstMu;
	old_b = &firstB;

	new_priors = &secondPriors;
	new_mu = &secondMu;
	new_b = &secondB;

	bool oldIsFirst = true;

	for (int i=0; i < numOfSamples; i++)
	{
		currentMixtureWeights[i] = new fp_type[numOfModels];
		tempMixtureWeights[i] = new fp_type[numOfModels];
	}

	int iter;

	for (iter=0; iter < max_iterations; iter++)
	{
		mexPrintf("Starting iteration #%d\n",iter);

		if (iter > 0)
		{
			for (int i=0; i < numOfSamples; i++)
			{
				for (int k=0; k<numOfModels; k++)
				{
					tempMixtureWeights[i][k] = currentMixtureWeights[i][k];
				}
			}
		}

		oldSamplesLogLikelihood = getSamplesMixtureLogLikelihood(numOfModels, numOfSamples, numOfDims, samples, *old_mu, *old_b, *old_priors);
	
		mexPrintf("Current Likelihood %f\n",oldSamplesLogLikelihood);
		mexEvalString("drawnow");

		computeCurrentMixtureWeightsPara(numOfSamples, numOfDims, numOfModels, samples, *old_mu, *old_b, *old_priors, currentMixtureWeights, totalMixtureWeights);
		
		fp_type diffMixtureWeightsNorm = 0.0;
			
		if (iter > 0)
		{
			for (int i=0; i < numOfSamples; i++)
			{
				for (int k=0; k<numOfModels; k++)
				{
					diffMixtureWeightsNorm = diffMixtureWeightsNorm + (tempMixtureWeights[i][k] - currentMixtureWeights[i][k]) * (tempMixtureWeights[i][k] - currentMixtureWeights[i][k]);
				}
			}

			diffMixtureWeightsNorm = sqrt(diffMixtureWeightsNorm);
			
			mexPrintf(", diffNorm %f",diffMixtureWeightsNorm);

			if (diffMixtureWeightsNorm < thres)
			{
				break;
			}
		}

		mexPrintf("\n");

		updateMixtureMuPara(numOfSamples, numOfDims, numOfModels, samples, currentMixtureWeights, totalMixtureWeights, *new_mu);
		updateMixtureBPara(numOfSamples, numOfDims, numOfModels, samples, currentMixtureWeights, *new_mu, totalMixtureWeights, *new_b);
		updateMixturePriors(numOfSamples, numOfModels, totalMixtureWeights, *new_priors);

		//newSamplesLogLikelihood = getSamplesMixtureLogLikelihood(numOfModels, numOfSamples, numOfDims, samples, *new_mu, *new_b, *new_priors);

		if (oldIsFirst)
		{
			old_priors = &secondPriors;
			old_mu = &secondMu;
			old_b = &secondB;

			new_priors = &firstPriors;
			new_mu = &firstMu;
			new_b = &firstB;

			oldIsFirst = false;
		}
		else
		{
			old_priors = &firstPriors;
			old_mu = &firstMu;
			old_b = &firstB;

			new_priors = &secondPriors;
			new_mu = &secondMu;
			new_b = &secondB;
			
			oldIsFirst = true;
		}

		oldSamplesLogLikelihood = newSamplesLogLikelihood;
	}

	for (int k=0; k < numOfModels; k++)
	{
		for (int d=0; d<numOfDims; d++)
		{
			out_final_mu[k][d] = old_mu[0][k][d];
			out_final_b[k][d] = old_b[0][k][d];
		}

		out_final_prior[k] = old_priors[0][k];
	}

	out_num_of_iter[0] = iter; 

	for (int i=0; i < numOfSamples; i++)
	{
		for (int k=0; k < numOfModels; k++)
		{
			out_final_mixture_weight[i][k] = currentMixtureWeights[i][k];
		}
	}
		
	delete[] firstPriors;
	delete[] secondPriors;
	delete[] totalMixtureWeights;

	for (int k=0; k < numOfModels; k++)
	{
		delete[] firstMu[k]; 
		delete[] secondMu[k];
		delete[] firstB[k];
		delete[] secondB[k];
	}

	delete[] firstMu;
	delete[] secondMu;
	delete[] firstB;
	delete[] secondB;

	for (int i=0; i < numOfSamples; i++)
	{
		delete[] currentMixtureWeights[i];
		delete[] tempMixtureWeights[i];
	}

	delete[] currentMixtureWeights;
	delete[] tempMixtureWeights;
}

