#ifndef FV_H
#define FV_H

#include "Hybrid.h"
#include "Util.h"

void computeFV(int numOfSamples, int numOfModels, int numOfDims, fp_type** samples, fp_type** m, fp_type** s, fp_type** mu, fp_type** sigma, bool** b, fp_type* priors, fp_type* out_fv, bool normDescCoeff, int numOfCores);

#endif