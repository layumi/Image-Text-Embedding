#ifndef FV_H
#define FV_H

#include "LMM.h"
#include "UtilLMM.h"

void computeFV(int numOfSamples, int numOfModels, int numOfDims, fp_type** samples, fp_type** mu, fp_type** b, fp_type* priors, fp_type* out_fv, bool normDescCoeff);


#endif