#ifndef UTILHYBRID_H
#define UTILHYBRID_H

typedef double fp_type; /* floating point type */

inline fp_type getUnivariateLaplaceLogPDF(fp_type x, fp_type m, fp_type s);
inline fp_type getUnivariateGaussianLogPDF(fp_type x, fp_type mu, fp_type sigma);

inline fp_type getMultivariateHybridLogPDF(int numOfDim, fp_type* x, fp_type* m, fp_type* s, fp_type* mu, fp_type* sigma, bool* b);
inline fp_type getMultivariateHybridPDF(int numOfDim, fp_type* x, fp_type* m, fp_type* s, fp_type* mu, fp_type* sigma, bool* b);

inline void getMixtureWeightsForSingleSample(int numOfModels, int numOfDims, fp_type* x, fp_type** m, fp_type** s, fp_type** mu, fp_type** sigma, bool** b, fp_type* priors, fp_type* mixtureWeightsOut);

fp_type getSamplesMixtureLogLikelihood(int numOfModels, int numOfSamples, int numOfDims, fp_type** x, fp_type** m, fp_type** s, fp_type** mu, fp_type** sigma, bool** b, fp_type* priors);

#endif
