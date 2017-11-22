#include "Util.h"
#include <math.h>
#include "float.h"

#define M_PI 3.14159265358979323846

inline fp_type getUnivariateLaplaceLogPDF(fp_type x, fp_type m, fp_type s)
{
	fp_type result = -1*log(2*s) - ((abs(x - m) / s));
	return result;
}

inline fp_type getUnivariateGaussianLogPDF(fp_type x, fp_type mu, fp_type sigma)
{
	fp_type result = -1*log(sqrt(sigma) * sqrt(2*M_PI)) - (((x-mu)*(x-mu)) / (2 * sigma));
	return result;
}


inline fp_type getMultivariateHybridLogPDF(int numOfDim, fp_type* x, fp_type* m, fp_type* s, fp_type* mu, fp_type* sigma, bool* b) 
{
	/* the assumption is that that the dimensions are independent */
	fp_type result = 0.0;

	for (int d=0; d<numOfDim; d++)
	{	
		if (b[d] == 1) // laplace
		{
			result = result + getUnivariateLaplaceLogPDF(x[d], m[d], s[d]);
		}
		else // gaussian
		{
			result = result + getUnivariateGaussianLogPDF(x[d], mu[d], sigma[d]);	
		}
	}

	return result;
}

inline fp_type getMultivariateHybridPDF(int numOfDim, fp_type* x, fp_type* m, fp_type* s, fp_type* mu, fp_type* sigma, bool* b) 
{
	return exp(getMultivariateHybridLogPDF(numOfDim, x, m, s, mu, sigma, b));
}

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

fp_type getSamplesMixtureLogLikelihood(int numOfModels, int numOfSamples, int numOfDims, fp_type** x, fp_type** m, fp_type** s, fp_type** mu, fp_type** sigma, bool** b, fp_type* priors)
{
	fp_type result = 0.0;

	for (int i=0; i < numOfSamples; i++)
	{
		fp_type currentSampleProb = 0.0;

		for (int k=0; k < numOfModels; k++)
		{
			currentSampleProb = currentSampleProb + priors[k] * getMultivariateHybridPDF(numOfDims, x[i], m[k], s[k], mu[k], sigma[k], b[k]);
		}

		#pragma omp critical
		{
			result = result + log(currentSampleProb);
		}
	}

	return result;
}
