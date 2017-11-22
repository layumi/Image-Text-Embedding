#include "FV.h"
#include "Util.h"
#include <cmath>
#include <omp.h>


void computeFV(int numOfSamples, int numOfModels, int numOfDims, fp_type** samples, fp_type** m, fp_type** s, fp_type** mu, fp_type** sigma, bool** b, fp_type* priors, fp_type* out_fv, bool normDescCoeff, int numOfCores)
{
	omp_set_num_threads(numOfCores);

	fp_type** currentMixtureWeights = new fp_type*[numOfSamples];
	fp_type* totalMixtureWeights = new fp_type[numOfModels];
	
	#pragma omp parallel for
	for (int i=0; i < numOfSamples; i++)
	{
		currentMixtureWeights[i] = new fp_type[numOfModels];
	}

	computeCurrentMixtureWeightsPara(numOfSamples, numOfDims, numOfModels, samples, m, s, mu, sigma, b, priors, currentMixtureWeights, totalMixtureWeights);
		
	#pragma omp parallel for
	for (int k=0; k < numOfModels; k++)
	{
		for (int d=0; d < numOfDims; d++)
		{
			fp_type currentAccumM = 0.0;
			fp_type currentAccumS = 0.0;
			fp_type currentAccumMu = 0.0;
			fp_type currentAccumSigma = 0.0;
			
			if (b[k][d]) // laplacian
			{
				for (int i=0; i < numOfSamples; i++)
				{
					if (samples[i][d] >= m[k][d])
					{
						currentAccumM = currentAccumM + currentMixtureWeights[i][k];
					}
					else
					{
						currentAccumM = currentAccumM - currentMixtureWeights[i][k];
					}

					currentAccumS = currentAccumS + currentMixtureWeights[i][k]*((std::abs(samples[i][d] - m[k][d]) / (s[k][d])) - 1);
				}

				currentAccumM = currentAccumM / s[k][d];
				currentAccumS = currentAccumS / s[k][d];

				if (normDescCoeff)
				{
					currentAccumM = (currentAccumM * (s[k][d])) / sqrt(numOfSamples * priors[k]);
					currentAccumS = (currentAccumS * (s[k][d])) / sqrt(numOfSamples * priors[k]);
				}

				out_fv[k*numOfDims + d] = currentAccumM;
				out_fv[numOfModels*numOfDims + k*numOfDims + d] = currentAccumS;
			}
			else // gaussian
			{
				for (int i=0; i < numOfSamples; i++)
				{
					currentAccumMu = currentAccumMu + currentMixtureWeights[i][k]*(samples[i][d] - mu[k][d]);
					currentAccumSigma = currentAccumSigma + currentMixtureWeights[i][k]*( ((samples[i][d] - mu[k][d])*(samples[i][d] - mu[k][d]) / sigma[k][d]) - 1);
				}
				
				currentAccumMu = currentAccumMu / sigma[k][d];
				currentAccumSigma = currentAccumSigma / sqrt(sigma[k][d]);
				
				if (normDescCoeff)
				{
					currentAccumMu = currentAccumMu * sqrt(sigma[k][d] / (numOfSamples * priors[k]));
					currentAccumSigma = currentAccumSigma * sqrt(sigma[k][d] / (2 * numOfSamples * priors[k]));
				}
				
				out_fv[k*numOfDims + d] = currentAccumMu;
				out_fv[numOfModels*numOfDims + k*numOfDims + d] = currentAccumSigma;

			}
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