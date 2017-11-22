#ifndef UTILHYBRID_H
#define UTILHYBRID_H

#include <cmath>
#include "float.h"

typedef double fp_type; /* floating point type */

inline fp_type getUnivariateLaplaceLogPDF(fp_type x, fp_type m, fp_type s);
inline fp_type getUnivariateGaussianLogPDF(fp_type x, fp_type mu, fp_type sigma);

inline fp_type getMultivariateHybridLogPDF(int numOfDim, fp_type* x, fp_type* m, fp_type* s, fp_type* mu, fp_type* sigma, bool* b);
inline fp_type getMultivariateHybridPDF(int numOfDim, fp_type* x, fp_type* m, fp_type* s, fp_type* mu, fp_type* sigma, bool* b);

inline void getMixtureWeightsForSingleSample(int numOfModels, int numOfDims, fp_type* x, fp_type** m, fp_type** s, fp_type** mu, fp_type** sigma, bool** b, fp_type* priors, fp_type* mixtureWeightsOut);


inline fp_type getUnivariateGaussianLogPDF(fp_type x, fp_type mu, fp_type sigma)
{
	fp_type result = -1*log(sqrt(sigma) * sqrt(2*M_PI)) - (((x-mu)*(x-mu)) / (2 * sigma));
	return result;
}

inline fp_type getUnivariateLaplaceLogPDF(fp_type x, fp_type m, fp_type s)
{
	fp_type result = -1*log(2*s) - ((std::abs(x - m) / s));
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

fp_type getSamplesMixtureLogLikelihood(int numOfModels, int numOfSamples, int numOfDims, fp_type** x, fp_type** m, fp_type** s, fp_type** mu, fp_type** sigma, bool** b, fp_type* priors);

#endif
