#include "UtilLMM.h"
#include <cmath>
#include "float.h"

fp_type getMixtureWeightsForSingleSample(int numOfModels, int numOfDims, fp_type* x, fp_type** mu, fp_type** b, fp_type* priors, fp_type* mixtureWeightsOut)
{
	fp_type maxExp = -DBL_MAX;

	for (int k=0; k<numOfModels; k++)
	{
		mixtureWeightsOut[k] = getMultivariateLaplaceLogPDF(numOfDims, x, mu[k], b[k]);

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
	
	return 0.0;
}

fp_type getSamplesMixtureLogLikelihood(int numOfModels, int numOfSamples, int numOfDims, fp_type** x, fp_type** mu, fp_type** b, fp_type* priors)
{
	fp_type result = 0.0;
	
	#pragma omp parallel for
	for (int i=0; i < numOfSamples; i++)
	{
		fp_type currentSampleProb = 0.0;

		for (int k=0; k < numOfModels; k++)
		{
			currentSampleProb = currentSampleProb + priors[k] * getMultivariateLaplacePDF(numOfDims, x[i], mu[k], b[k]);
		}

		#pragma omp critical
		{
			result = result + log(currentSampleProb);
		}
	}

	return result;
}
