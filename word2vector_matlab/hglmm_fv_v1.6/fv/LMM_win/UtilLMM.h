#ifndef UTILLMM_H
#define UTILLMM_H

typedef double fp_type; /* floating point type */

inline fp_type getUnivariateLaplaceLogPDF(fp_type x, fp_type mu, fp_type b);
inline fp_type getMultivariateLaplaceLogPDF(int numOfDim, fp_type* x, fp_type* mu, fp_type* b);
inline fp_type getMultivariateLaplacePDF(int numOfDim, fp_type* x, fp_type* mu, fp_type* b); 

fp_type getMixtureWeightsForSingleSample(int numOfModels, int numOfDims, fp_type* x, fp_type** mu, fp_type** b, fp_type* priors, fp_type* mixtureWeightsOut);


fp_type getSamplesMixtureLogLikelihood(int numOfModels, int numOfSamples, int numOfDims, fp_type** x, fp_type** mu, fp_type** b, fp_type* priors);

#endif
