#include "Util.h"

#ifndef Hybrid_H
#define Hybrid_H

void lmmEM(int numOfSamples, int numOfModels, int numOfDims, int max_iterations, fp_type** samples, fp_type** init_m, fp_type** init_s, fp_type** init_mu, fp_type** init_sigma, bool** init_b, fp_type* init_priors, fp_type** out_final_m, fp_type** out_final_s, fp_type** out_final_mu, fp_type** out_final_sigma, bool** out_final_b, fp_type* out_final_prior, int* out_num_of_iter, fp_type** out_final_mixture_weight, int numOfCores, fp_type min_s, fp_type min_sigma);
void computeCurrentMixtureWeightsPara(int numOfSamples, int numOfDims, int numOfModels, fp_type** samples, fp_type** m, fp_type** s, fp_type** mu, fp_type** sigma, bool** b, fp_type* priors, fp_type** outMixtureWeights, fp_type* outTotalMixtureWeight);
#endif 