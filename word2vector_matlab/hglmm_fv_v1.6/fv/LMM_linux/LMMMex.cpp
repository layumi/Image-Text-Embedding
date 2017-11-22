#include "mex.h" 
#include "LMM.h"
#include "UtilLMM.h"

#define NUM_OF_INPUTS 7
#define SAMPLES_INPUT_IDX 0
#define INIT_MU_INPUT_IDX 1
#define INIT_B_INPUT_IDX 2
#define INIT_PRIOR_INPUT_IDX 3
#define MAX_ITERATIONS_INPUT_IDX 4
#define THRESHOLD_INPUT_IDX 5
#define NUMOFCORES_INPUT_IDX 6

#define MU_OUTPUT_IDX 0
#define B_OUTPUT_IDX 1
#define PRIORS_OUTPUT_IDX 2
#define SAMPLES_WEIGHTS_OUTPUT_IDX 3

void mexFunction(int nlhs, mxArray *plhs[], /* Output variables */
	int nrhs, const mxArray *prhs[]) /* Input variables */
{
	if(nrhs < (NUM_OF_INPUTS - 1)) /* Check the number of arguments */
		mexErrMsgTxt("Wrong number of input arguments.");
	
	int numOfCores = 1;
	if (nrhs == (NUM_OF_INPUTS - 1))
	{
		mexPrintf("Number Of Cores Was not supplied, using only 1\n"); 
	}
	else
	{
		double* cores_ptr = mxGetPr(prhs[NUMOFCORES_INPUT_IDX]);
		double cores = cores_ptr[0];
		numOfCores = cores;

		mexPrintf("Number Of Cores is %d\n", numOfCores); 
	}

	double* samples = mxGetPr(prhs[SAMPLES_INPUT_IDX]);
	int numOfSamples = mxGetM(prhs[SAMPLES_INPUT_IDX]);
	int numOfDims = mxGetN(prhs[SAMPLES_INPUT_IDX]);

	double* mu = mxGetPr(prhs[INIT_MU_INPUT_IDX]);
	int classes = mxGetM(prhs[INIT_MU_INPUT_IDX]);

	int N = mxGetN(prhs[INIT_MU_INPUT_IDX]);
	if (N != numOfDims)
	{
		mexErrMsgTxt("Wrong number of dimensions for mu init.");
	}

	double* b = mxGetPr(prhs[INIT_B_INPUT_IDX]);
	int M = mxGetM(prhs[INIT_B_INPUT_IDX]);
	N = mxGetN(prhs[INIT_B_INPUT_IDX]);

	if (M != classes)
	{
		mexErrMsgTxt("Wrong number of classes for b init.");
	}

	if (N != numOfDims)
	{
		mexErrMsgTxt("Wrong number of dimensions for b init.");
	}

	double* priors = mxGetPr(prhs[INIT_PRIOR_INPUT_IDX]);
	M = mxGetM(prhs[INIT_PRIOR_INPUT_IDX]);
	
	if (M != classes)
	{
		mexErrMsgTxt("Wrong number of classes for prior init.");
	}

	fp_type** samples_for_algo;
	fp_type** mu_for_algo;
	fp_type** b_for_algo;
	fp_type* prior_for_algo;

	fp_type** out_mu;
	fp_type** out_b;
	fp_type* out_priors;

	samples_for_algo = new fp_type*[numOfSamples];
	for (int i=0; i < numOfSamples; i++)
	{
		samples_for_algo[i] = new fp_type[numOfDims];
		for (int d=0; d < numOfDims; d++)
		{
			samples_for_algo[i][d] = samples[d*numOfSamples + i];
		}
	}

	mu_for_algo = new fp_type*[classes];
	out_mu = new fp_type*[classes];
	for (int i=0; i < classes; i++)
	{
		mu_for_algo[i] = new fp_type[numOfDims];
		out_mu[i] = new fp_type[numOfDims];

		for (int d=0; d < numOfDims; d++)
		{
			mu_for_algo[i][d] = mu[d*classes + i];
			out_mu[i][d] = 0.0;
		}
	}

	b_for_algo = new fp_type*[classes];
	out_b = new fp_type*[classes];
	for (int i=0; i < classes; i++)
	{
		b_for_algo[i] = new fp_type[numOfDims];
		out_b[i] = new fp_type[numOfDims];

		for (int d=0; d < numOfDims; d++)
		{
			b_for_algo[i][d] = b[d*classes + i];
			out_b[i][d] = 0.0;
		}
	}

	prior_for_algo = new fp_type[classes];
	out_priors = new fp_type[classes];
	
	for (int i=0; i < classes; i++)
	{
		prior_for_algo[i] = priors[i];
		out_priors[i] = 0.0;
	}

	double* maxNumOfIterations_ptr =  mxGetPr(prhs[MAX_ITERATIONS_INPUT_IDX]);
	int maxNumOfIterations = maxNumOfIterations_ptr[0];

	double* threshold_ptr = mxGetPr(prhs[THRESHOLD_INPUT_IDX]);
	double threshold = threshold_ptr[0];
	
	int outNumOfIterations = 0.0;
	
	fp_type** out_mixture_weights = new fp_type*[numOfSamples];

	for (int i=0; i<numOfSamples; i++)
	{
		out_mixture_weights[i] = new fp_type[classes];
		
		for (int k=0; k < classes; k++)
		{
			out_mixture_weights[i][k] = 0.0;
		}
	}

	lmmEM(numOfSamples, classes, numOfDims, maxNumOfIterations, threshold, samples_for_algo, mu_for_algo, b_for_algo, prior_for_algo, out_mu, out_b, out_priors, &outNumOfIterations, out_mixture_weights, numOfCores);
	outNumOfIterations++;

	plhs[SAMPLES_WEIGHTS_OUTPUT_IDX] = mxCreateDoubleMatrix(numOfSamples, classes, mxREAL); 
	double* samplesWeightsMexOut = mxGetPr(plhs[SAMPLES_WEIGHTS_OUTPUT_IDX]); 

	plhs[MU_OUTPUT_IDX] = mxCreateDoubleMatrix(classes, numOfDims, mxREAL); 
	double* muMexOut = mxGetPr(plhs[MU_OUTPUT_IDX]); 

	plhs[B_OUTPUT_IDX] = mxCreateDoubleMatrix(classes, numOfDims, mxREAL); 
	double* bMexOut = mxGetPr(plhs[B_OUTPUT_IDX]); 

	plhs[PRIORS_OUTPUT_IDX] = mxCreateDoubleMatrix(classes, 1, mxREAL); 
	double* priorsMexOut = mxGetPr(plhs[PRIORS_OUTPUT_IDX]); 

	for (int i=0; i < numOfSamples; i++)
	{
		for (int k=0; k < classes; k++)
		{
			samplesWeightsMexOut[k*numOfSamples + i] = out_mixture_weights[i][k];
		}
	}

	for (int k=0; k < classes; k++)
	{
		for (int d=0; d < numOfDims; d++)
		{
			muMexOut[d*classes + k] = out_mu[k][d];
			bMexOut[d*classes + k] = out_b[k][d];
		}

		priorsMexOut[k] = out_priors[k];
	}

	return;
}