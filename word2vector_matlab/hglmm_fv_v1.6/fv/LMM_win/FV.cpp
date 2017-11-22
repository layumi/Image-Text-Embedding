#include "FV.h"
#include "UtilLMM.h"
#include <math.h>

#include <omp.h>

#define MAX_CORES 6

void computeFV(int numOfSamples, int numOfModels, int numOfDims, fp_type** samples, fp_type** mu, fp_type** b, fp_type* priors, fp_type* out_fv, bool normDescCoeff)
{
	omp_set_num_threads(MAX_CORES);

	fp_type** currentMixtureWeights = new fp_type*[numOfSamples];
	fp_type* totalMixtureWeights = new fp_type[numOfModels];
	
	#pragma omp parallel for
	for (int i=0; i < numOfSamples; i++)
	{
		currentMixtureWeights[i] = new fp_type[numOfModels];
	}

	computeCurrentMixtureWeightsPara(numOfSamples, numOfDims, numOfModels, samples, mu, b, priors, currentMixtureWeights, totalMixtureWeights);

	#pragma omp parallel for
	for (int k=0; k < numOfModels; k++)
	{
		for (int d=0; d < numOfDims; d++)
		{
			fp_type currentAccumMu = 0.0;
			fp_type currentAccumB = 0.0;

			for (int i=0; i < numOfSamples; i++)
			{
				if (samples[i][d] >= mu[k][d])
				{
					currentAccumMu = currentAccumMu + currentMixtureWeights[i][k];
				}
				else
				{
					currentAccumMu = currentAccumMu - currentMixtureWeights[i][k];
				}

				currentAccumB = currentAccumB + currentMixtureWeights[i][k]*((abs(samples[i][d] - mu[k][d]) / (b[k][d])) - 1);
			}

			currentAccumMu = currentAccumMu / b[k][d];
			currentAccumB = currentAccumB / b[k][d];

			if (normDescCoeff)
			{
				currentAccumMu = (currentAccumMu * (b[k][d])) / sqrt(numOfSamples * priors[k]);
				currentAccumB = (currentAccumB * (b[k][d])) / sqrt(numOfSamples * priors[k]);
			}

			out_fv[k*numOfDims + d] = currentAccumMu;
			out_fv[numOfModels*numOfDims + k*numOfDims + d] = currentAccumB;
		}
	}

	#pragma omp parallel for
	for (int i=0; i < numOfSamples; i++)
	{
		delete[] currentMixtureWeights[i];
	}

	delete[] totalMixtureWeights;
	delete[] currentMixtureWeights;
}