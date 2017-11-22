#include "UtilLMM.h"

#ifndef LMM_H
#define LMM_H


void computeCurrentMixtureWeights(int numOfSamples, int numOfDims, int numOfModels, fp_type** samples, fp_type** mu, fp_type** b, fp_type* priors, fp_type** outMixtureWeights, fp_type* outTotalMixtureWeight);
void lmmEM(int numOfSamples, int numOfModels, int numOfDims, int max_iterations, fp_type thres, fp_type** samples, fp_type** init_mu, fp_type** init_b, fp_type* init_priors, fp_type** out_final_mu, fp_type** out_final_b, fp_type* out_final_prior, int* out_num_of_iter, fp_type** out_final_mixture_weight, int numOfCores);
void computeCurrentMixtureWeightsPara(int numOfSamples, int numOfDims, int numOfModels, fp_type** samples, fp_type** mu, fp_type** b, fp_type* priors, fp_type** outMixtureWeights, fp_type* outTotalMixtureWeight);

#endif 