#ifndef UTILLMM_H
#define UTILLMM_H

#include <cmath>
#include "float.h"

typedef double fp_type; /* floating point type */

inline fp_type getUnivariateLaplaceLogPDF(fp_type x, fp_type mu, fp_type b)
{
	fp_type result = -1*log(2*b) - ((std::abs(x - mu) / b));
	return result;
}

inline fp_type getMultivariateLaplaceLogPDF(int numOfDim, fp_type* x, fp_type* mu, fp_type* b) 
{
	/* the assumption is that that the dimensions are independent */
	fp_type result = 0.0;

	for (int d=0; d<numOfDim; d++)
	{
		result = result + getUnivariateLaplaceLogPDF(x[d], mu[d], b[d]);
	}

	return result;
}

inline fp_type getMultivariateLaplacePDF(int numOfDim, fp_type* x, fp_type* mu, fp_type* b) 
{
	return exp(getMultivariateLaplaceLogPDF(numOfDim, x, mu, b));
}


fp_type getMixtureWeightsForSingleSample(int numOfModels, int numOfDims, fp_type* x, fp_type** mu, fp_type** b, fp_type* priors, fp_type* mixtureWeightsOut);


fp_type getSamplesMixtureLogLikelihood(int numOfModels, int numOfSamples, int numOfDims, fp_type** x, fp_type** mu, fp_type** b, fp_type* priors);

#endif
